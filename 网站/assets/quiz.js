/*
 * 背题库：每天按日期种子随机抽题，答案按需注入。
 * 题库数据由 网站/build_quiz.py 在构建时生成，页面本身不需要后端。
 */
(function () {
  "use strict";

  var dataPromise = null;

  function escapeHtml(value) {
    return String(value == null ? "" : value).replace(/[&<>"']/g, function (ch) {
      return {
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;",
      }[ch];
    });
  }

  function localDateKey() {
    var d = new Date();
    var month = String(d.getMonth() + 1).padStart(2, "0");
    var day = String(d.getDate()).padStart(2, "0");
    return d.getFullYear() + "-" + month + "-" + day;
  }

  // FNV-1a：把“日期 + 分类 + 批次”变成稳定的 32 位种子。
  function hashSeed(value) {
    var hash = 2166136261;
    for (var i = 0; i < value.length; i += 1) {
      hash ^= value.charCodeAt(i);
      hash = Math.imul(hash, 16777619);
    }
    return hash >>> 0;
  }

  function mulberry32(seed) {
    return function () {
      var t = (seed += 0x6d2b79f5);
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function seededShuffle(list, seed) {
    var result = list.slice();
    var random = mulberry32(seed);
    for (var i = result.length - 1; i > 0; i -= 1) {
      var j = Math.floor(random() * (i + 1));
      var tmp = result[i];
      result[i] = result[j];
      result[j] = tmp;
    }
    return result;
  }

  function typesetMath(target, attempt) {
    if (window.MathJax && typeof window.MathJax.typesetPromise === "function") {
      window.MathJax.typesetPromise(target ? [target] : undefined).catch(function (error) {
        console.warn("[MathJax] 背题库排版失败：", error);
      });
      return;
    }
    attempt = attempt || 0;
    if (attempt < 60) {
      window.setTimeout(function () {
        typesetMath(target, attempt + 1);
      }, 150);
    }
  }

  function loadData(url) {
    if (!dataPromise) {
      dataPromise = fetch(url, { cache: "no-cache" })
        .then(function (response) {
          if (!response.ok) {
            throw new Error("HTTP " + response.status);
          }
          return response.json();
        })
        .catch(function (error) {
          dataPromise = null;
          throw error;
        });
    }
    return dataPromise;
  }

  function initQuiz(app) {
    var countSelect = app.querySelector("#quiz-count");
    var categorySelect = app.querySelector("#quiz-category");
    var refreshButton = app.querySelector("#quiz-refresh");
    var todayButton = app.querySelector("#quiz-today");
    var revealAllButton = app.querySelector("#quiz-reveal-all");
    var hideAllButton = app.querySelector("#quiz-hide-all");
    var dateBadge = app.querySelector("#quiz-date");
    var poolLabel = app.querySelector("#quiz-pool");
    var progressLabel = app.querySelector("#quiz-progress");
    var loading = app.querySelector("#quiz-loading");
    var list = app.querySelector("#quiz-list");
    var dataUrl = app.getAttribute("data-src");

    var state = {
      data: null,
      date: localDateKey(),
      category: "all",
      count: parseInt(countSelect.value, 10) || 10,
      batch: 0,
      selected: [],
      revealed: {},
    };

    function categoryName(id) {
      for (var i = 0; i < state.data.categories.length; i += 1) {
        if (state.data.categories[i].id === id) {
          return state.data.categories[i].name;
        }
      }
      return id;
    }

    function currentPool() {
      if (state.category === "all") {
        return state.data.questions;
      }
      return state.data.questions.filter(function (item) {
        return item.category === state.category;
      });
    }

    function seedString() {
      var batch = state.batch === 0 ? "daily" : "batch-" + state.batch;
      return [state.date, state.category, batch, "quiz-v1"].join("|");
    }

    function updateMeta(poolSize) {
      var mode = state.batch === 0 ? "今日随机" : "换一批 #" + state.batch;
      dateBadge.textContent = state.date + " · " + mode;
      var suffix = state.selected.length < state.count ? "（题库不足）" : "";
      poolLabel.textContent =
        "当前分类 " + poolSize + " 道 · 本次 " + state.selected.length + " 道" + suffix;
    }

    function updateProgress() {
      var total = state.selected.length;
      var seen = Object.keys(state.revealed).length;
      progressLabel.textContent = "已看 " + seen + " / " + total;
      revealAllButton.disabled = total === 0 || seen >= total;
      hideAllButton.disabled = seen === 0;
    }

    function fillAnswer(card, index) {
      var answer = card.querySelector(".quiz-answer");
      if (answer.getAttribute("data-loaded") === "1") {
        return false;
      }
      answer.innerHTML =
        '<div class="quiz-answer-label">参考答案</div>' +
        state.selected[index].answerHtml;
      answer.setAttribute("data-loaded", "1");
      return true;
    }

    function renderQuestions(poolSize) {
      if (!state.selected.length) {
        list.innerHTML = '<div class="quiz-empty">这个分类暂时没有题目。</div>';
        return;
      }

      var html = state.selected.map(function (item, index) {
        var answerId = "quiz-answer-" + index;
        return (
          '<article class="quiz-card" data-index="' +
          index +
          '" data-id="' +
          escapeHtml(item.id) +
          '">' +
          '<div class="quiz-card__head">' +
          '<span class="quiz-index">' +
          String(index + 1).padStart(2, "0") +
          "</span>" +
          '<div class="quiz-tags">' +
          '<span class="quiz-tag quiz-tag--category">' +
          escapeHtml(item.categoryName) +
          "</span>" +
          '<span class="quiz-tag">' +
          escapeHtml(item.topic) +
          "</span>" +
          "</div>" +
          "</div>" +
          '<div class="quiz-question">' +
          item.questionHtml +
          "</div>" +
          '<div class="quiz-card__actions">' +
          '<button type="button" class="quiz-button quiz-toggle" data-index="' +
          index +
          '" aria-expanded="false" aria-controls="' +
          answerId +
          '">显示答案</button>' +
          '<span class="quiz-source">来源：' +
          escapeHtml(item.sourceTitle) +
          "</span>" +
          "</div>" +
          '<div class="quiz-answer" id="' +
          answerId +
          '" data-loaded="0" hidden></div>' +
          "</article>"
        );
      });
      list.innerHTML = html.join("");
      updateMeta(poolSize);
      updateProgress();
      typesetMath(list);
    }

    function pick() {
      if (!state.data) {
        return;
      }
      var pool = currentPool();
      state.selected = seededShuffle(pool, hashSeed(seedString())).slice(
        0,
        Math.min(state.count, pool.length)
      );
      state.revealed = {};
      renderQuestions(pool.length);
    }

    function setAnswer(index, reveal) {
      var card = list.querySelector('.quiz-card[data-index="' + index + '"]');
      if (!card) {
        return;
      }
      var answer = card.querySelector(".quiz-answer");
      var button = card.querySelector(".quiz-toggle");
      var wasRevealed = !answer.hidden;
      if (reveal === wasRevealed) {
        return;
      }

      if (reveal) {
        var newlyFilled = fillAnswer(card, index);
        answer.hidden = false;
        card.classList.add("is-revealed");
        button.textContent = "隐藏答案";
        button.setAttribute("aria-expanded", "true");
        state.revealed[index] = true;
        if (newlyFilled) {
          typesetMath(answer);
        }
      } else {
        answer.hidden = true;
        card.classList.remove("is-revealed");
        button.textContent = "显示答案";
        button.setAttribute("aria-expanded", "false");
        delete state.revealed[index];
      }
      updateProgress();
    }

    function setAllAnswers(reveal) {
      if (!state.selected.length) {
        return;
      }
      for (var i = 0; i < state.selected.length; i += 1) {
        var card = list.querySelector('.quiz-card[data-index="' + i + '"]');
        if (!card) {
          continue;
        }
        var answer = card.querySelector(".quiz-answer");
        var button = card.querySelector(".quiz-toggle");
        if (reveal) {
          fillAnswer(card, i);
          answer.hidden = false;
          card.classList.add("is-revealed");
          button.textContent = "隐藏答案";
          button.setAttribute("aria-expanded", "true");
          state.revealed[i] = true;
        } else {
          answer.hidden = true;
          card.classList.remove("is-revealed");
          button.textContent = "显示答案";
          button.setAttribute("aria-expanded", "false");
          delete state.revealed[i];
        }
      }
      updateProgress();
      if (reveal) {
        typesetMath(list);
      }
    }

    function fillCategorySelect() {
      var options = state.data.categories.map(function (category) {
        return (
          '<option value="' +
          escapeHtml(category.id) +
          '">' +
          escapeHtml(category.name) +
          "（" +
          category.count +
          "）</option>"
        );
      });
      categorySelect.innerHTML = options.join("");
      categorySelect.value = state.category;
    }

    countSelect.addEventListener("change", function () {
      state.count = parseInt(countSelect.value, 10) || 10;
      pick();
    });

    categorySelect.addEventListener("change", function () {
      state.category = categorySelect.value || "all";
      // 切分类后回到该分类的“今日随机”，避免沿用上一批的批次号。
      state.batch = 0;
      pick();
    });

    refreshButton.addEventListener("click", function () {
      state.date = localDateKey();
      state.batch += 1;
      pick();
    });

    todayButton.addEventListener("click", function () {
      state.date = localDateKey();
      state.batch = 0;
      pick();
    });

    revealAllButton.addEventListener("click", function () {
      setAllAnswers(true);
    });

    hideAllButton.addEventListener("click", function () {
      setAllAnswers(false);
    });

    list.addEventListener("click", function (event) {
      var button = event.target.closest(".quiz-toggle");
      if (!button || !list.contains(button)) {
        return;
      }
      var index = parseInt(button.getAttribute("data-index"), 10);
      var card = button.closest(".quiz-card");
      var answer = card && card.querySelector(".quiz-answer");
      if (answer) {
        setAnswer(index, answer.hidden);
      }
    });

    loadData(dataUrl)
      .then(function (data) {
        if (!app.isConnected) {
          return;
        }
        if (!data || !Array.isArray(data.questions) || !data.questions.length) {
          throw new Error("题库数据为空");
        }
        state.data = data;
        state.category = "all";
        state.count = parseInt(countSelect.value, 10) || 10;
        fillCategorySelect();
        loading.hidden = true;
        pick();
      })
      .catch(function (error) {
        if (!app.isConnected) {
          return;
        }
        loading.hidden = true;
        list.innerHTML =
          '<div class="quiz-error"><strong>题库加载失败</strong>' +
          "<p>请刷新页面重试，或直接阅读 " +
          '<a href="../06-面试题库/04-真实面经高频题/">真实面经高频题</a>。</p>' +
          '<p class="quiz-muted">' +
          escapeHtml(error && error.message ? error.message : "未知错误") +
          "</p></div>";
        console.warn("[quiz] 加载失败：", error);
      });
  }

  function boot() {
    var app = document.getElementById("quiz-app");
    if (!app || app.getAttribute("data-quiz-init") === "1") {
      return;
    }
    app.setAttribute("data-quiz-init", "1");
    initQuiz(app);
  }

  if (typeof document$ !== "undefined" && document$.subscribe) {
    document$.subscribe(boot);
  } else if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
