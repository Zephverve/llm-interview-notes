---
title: 背题库
description: 每天从大模型面试题库随机抽题，选择题目数量，先作答再展开参考答案。
---

# 背题库

<p class="quiz-lead">每天从 <strong>算法岗、应用岗、系统设计、真实面经、项目深挖、手撕代码</strong> 中随机抽题。先自己回答，再展开参考答案；同一天、同一分类下题目固定，点“换一批”可以重新抽。</p>

<div id="quiz-app" class="quiz-app" data-src="../assets/quiz-data.json">
  <div class="quiz-toolbar" role="region" aria-label="背题设置">
    <div class="quiz-controls">
      <label class="quiz-field">
        <span>题目数量</span>
        <select id="quiz-count">
          <option value="5">5 道</option>
          <option value="10" selected>10 道</option>
          <option value="20">20 道</option>
          <option value="30">30 道</option>
        </select>
      </label>
      <label class="quiz-field">
        <span>题目分类</span>
        <select id="quiz-category">
          <option value="all">全部题库</option>
        </select>
      </label>
      <div class="quiz-actions">
        <button type="button" id="quiz-refresh" class="quiz-button quiz-button--primary">换一批</button>
        <button type="button" id="quiz-today" class="quiz-button">今日随机</button>
      </div>
    </div>

    <div class="quiz-meta">
      <span id="quiz-date" class="quiz-badge">今日随机</span>
      <span id="quiz-pool" class="quiz-muted"></span>
      <span id="quiz-progress" class="quiz-muted">已看 0 / 0</span>
    </div>

    <div class="quiz-answer-actions">
      <button type="button" id="quiz-reveal-all" class="quiz-link-button">显示全部答案</button>
      <button type="button" id="quiz-hide-all" class="quiz-link-button">隐藏全部答案</button>
    </div>
  </div>

  <div id="quiz-loading" class="quiz-loading">正在加载题库…</div>
  <div id="quiz-list" class="quiz-list" aria-live="polite"></div>

  <noscript>
    <p class="quiz-error">背题库需要启用 JavaScript。你仍然可以直接阅读
      <a href="../06-面试题库/04-真实面经高频题/">真实面经高频题</a>。
    </p>
  </noscript>
</div>
