---
title: Home
hide:
  - navigation
  - toc
---

<p align="center"> <img src="_assets/logo.svg" alt="Logo" width="120em" /> </p>

# Inflation Forecasting {: align="center" }

<div class="grid cards" markdown>

-   ## Project

    Forecast next-month CPI month-over-month inflation using macro-financial indicators, official economic texts, and deep learning models trained from scratch.

    **Student**: Oğuzhan Özkaya

    **Instructor**: Şafak Özden

    _ADA 447 Introduction to Deep Learning - TED University_

-   ## Overview

    This project forecasts CPI MoM for month t+1 using only information available at the end of month t. The main challenge is not only model accuracy, but also correct time alignment. CPI releases, macro-financial indicators, and text documents must be filtered by publication date so future information cannot leak into earlier forecasts.

    The modeling pipeline will compare simple baselines, classical machine learning baselines, numeric deep learning models, text encoders trained from scratch, and a final fusion model that combines numeric and text representations.

</div>

<div class="grid cards" markdown>

-   ## Objective

    Build a reproducible term project that can download data, construct a leakage-safe monthly dataset, train baseline and deep learning models, evaluate them chronologically, and generate article-ready outputs.

-   ## Approach

    Combine numeric time-series features with a text branch that learns inflation-pressure representations from central bank publications and economic news without external pretrained models.

-   ## Concepts

    Time-series forecasting

    NLP from scratch

    CNNs, recurrent models

    Transformer encoders

    multimodal fusion

    optimization

    interpretability

    reproducible deployment-oriented workflows.

</div>

---

## Article
