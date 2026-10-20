/*
 * Material for MkDocs 的 instant navigation 只替换正文，
 * 不会自动重新排版新页面里的公式。这里在每次导航后调用 MathJax。
 */
(function () {
  "use strict";

  var retryTimer = null;
  var retryCount = 0;

  function typesetNow() {
    if (retryTimer) {
      clearTimeout(retryTimer);
      retryTimer = null;
    }
    if (!window.MathJax || typeof window.MathJax.typesetPromise !== "function") {
      return false;
    }
    window.MathJax.typesetPromise().catch(function (error) {
      console.warn("[MathJax] typeset failed:", error);
    });
    return true;
  }

  function scheduleTypeset() {
    retryCount = 0;
    if (typesetNow()) {
      return;
    }
    (function retry() {
      if (typesetNow()) {
        return;
      }
      retryCount += 1;
      if (retryCount <= 60) {
        retryTimer = setTimeout(retry, 100);
      }
    })();
  }

  /*
   * MkDocs Material 的搜索框只在 keyup 时查询。手机中文输入法、
   * 粘贴、语音输入有时只触发 input，不触发 keyup，这里补一个兜底。
   */
  function wireSearchFallback() {
    document.querySelectorAll("input.md-search__input").forEach(function (input) {
      if (input.dataset.llmSearchWired === "1") {
        return;
      }
      input.dataset.llmSearchWired = "1";
      var lastValue = input.value;
      var lastKeyup = 0;
      var timer = null;

      input.addEventListener("keyup", function () {
        lastKeyup = Date.now();
      });

      input.addEventListener("input", function () {
        if (input.value === lastValue) {
          return;
        }
        lastValue = input.value;
        if (timer) {
          clearTimeout(timer);
        }
        timer = setTimeout(function () {
          // 桌面端真实 keyup 已经触发过查询，避免重复查询。
          if (Date.now() - lastKeyup > 200) {
            input.dispatchEvent(
              new KeyboardEvent("keyup", { bubbles: true, key: "Unidentified" })
            );
          }
        }, 160);
      });
    });
  }

  function markScrollableTables() {
    document.querySelectorAll(".md-typeset__scrollwrap").forEach(function (wrap) {
      var scrollable = wrap.scrollWidth > wrap.clientWidth + 2;
      var previous = wrap.previousElementSibling;
      var hasHint =
        previous &&
        previous.classList &&
        previous.classList.contains("table-scroll-hint");

      if (scrollable && !hasHint) {
        var hint = document.createElement("div");
        hint.className = "table-scroll-hint";
        hint.textContent = "表格可左右滑动 →";
        wrap.parentNode.insertBefore(hint, wrap);
      } else if (!scrollable && hasHint) {
        previous.remove();
      }
    });
  }

  function onPageReady() {
    scheduleTypeset();
    wireSearchFallback();
    // 等表格完成布局后再判断是否需要横向滑动提示
    setTimeout(markScrollableTables, 350);
  }

  if (typeof document$ !== "undefined" && document$.subscribe) {
    document$.subscribe(onPageReady);
  } else if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", onPageReady);
  } else {
    onPageReady();
  }
})();
