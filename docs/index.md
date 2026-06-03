---
title: Home
hide:
  - navigation
  - toc
---

# Football In-Play Prediction {: align="center" }

<div class="grid cards" markdown>

-   ## Project

    [**Medium Article**](https://oguzhanozkaya.github.io/football-inplay-prediction/)

    [:fontawesome-regular-file-lines: **Presentation**](https://oguzhanozkaya.github.io/football-inplay-prediction/)

    Predicting football match outcomes at minute 45 using deep learning.

    **Dataset**: [ESPN Soccer Data](https://www.kaggle.com/datasets/excel4soccer/espn-soccer-data)

    **Student**: Oğuzhan Özkaya

    **Instructor**: Şafak Özden

    _ADA 447 Introduction to Deep Learning - TED University_

</div>

<div class="grid cards" markdown>

-   ## Overview

    This project predicts whether the final result will be a home win, draw, or away win using only match information available through minute 45. The core challenge is time alignment: commentary, plays, key events, and lineup-derived inputs must be sliced so later match information cannot leak into the in-play prediction.

-   ## Objective

    Build a reproducible command-driven pipeline that validates local ESPN Soccer raw data, constructs leakage-safe match windows, trains one hybrid neural classifier, evaluates chronological splits, and generates report-ready outputs.

-   ## Approach

    Combine text commentary and numerical event features inside one raw-PyTorch architecture: a TextCNN encodes each text window, numeric features are projected per window, fused vectors are passed through a GRU, and the final hidden state predicts home/draw/away.

</div>
