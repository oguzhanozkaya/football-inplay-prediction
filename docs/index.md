---
title: Home
hide:
  - navigation
  - toc
---

<p align="center"> <img src="_assets/logo.svg" alt="Logo" width="120em" /> </p>

<h1 align="center" style="margin: 0;"> Football In-Play Prediction </h1>

<h3 align="center" style="margin: 0.6em;">
    Predicting football match outcomes at minute 45 using deep learning.
</h3>

<h3 align="center" style="margin: 1em;">
    <a href="https://oguzhanozkaya.github.io/football-inplay-prediction/">Medium Article</a> - <a href="https://oguzhanozkaya.github.io/football-inplay-prediction/">Presentation</a>
</h3>

---

<div class="grid cards" markdown>

-   ## Project

    This project predicts whether the final result will be a home win, draw, or away win using only match information available through minute 45. The core challenge is time alignment: commentary, plays, key events, and lineup-derived inputs must be sliced so later match information cannot leak into the in-play prediction.

    **Dataset**: [ESPN Soccer Data](https://www.kaggle.com/datasets/excel4soccer/espn-soccer-data)

    **Student**: Oğuzhan Özkaya

    **Instructor**: Şafak Özden

    _ADA 447 Introduction to Deep Learning - TED University_

</div>

<div class="grid cards" markdown>

-   ## Objective

    Build a reproducible command-driven pipeline that downloads or validates ESPN Soccer raw data, constructs leakage-safe first-half match features, trains one hybrid neural classifier, evaluates league-aware chronological splits, and generates report-ready outputs.

-   ## Approach

    Combine text commentary and numerical event features inside one raw-PyTorch architecture: a TextCNN encodes the first-half text, an MLP encodes first-half numeric features, and a fusion classifier predicts home/draw/away.

</div>
