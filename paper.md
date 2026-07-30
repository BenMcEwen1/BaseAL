---
title: 'BaseAL: An Open-Source Evaluation Framework for Active Learning'
tags:
  - Active Learning
  - Data Challenges
  - Evaluation
  - Validation
authors:
  - name: Ben J. McEwen
    orcid: 0000-0002-0869-7717
    equal-contrib: false
    affiliation: 1 # (Multiple affiliations must be quoted)
  - name: Shiqi Zhang
    orcid: 0009-0008-6012-8088
    equal-contrib: false # (This is how you can denote equal contributions between multiple authors)
    affiliation: 2
affiliations:
 - name: Department of Theoretical and Computational Ecology, Institute for Biodiversity and Ecosystem Dynamics (IBED), University of Amsterdam, the Netherlands
   index: 1
 - name: Faculty of Information Technology and Communication Sciences, Tampere University, Finland
   index: 2
date: 29 June 2026
bibliography: paper.bib
---

<!-- Public development began on November 8th 2025 -->

# Summary

Machine learning models depend on the quality and quantity of labelled training data, yet acquiring labels is often expensive and requires specialist expertise. Active learning (AL) addresses this bottleneck through an iterative process of sample selection and label acquisition which aims to optimise a learning objective; improving model performance metrics with fewer labelled samples [@settles2012active]. Despite continued methodological development, standardised frameworks for evaluating AL methods, that consider broader computational and validation related challenges, remain limited.

[BaseAL](https://github.com/BenMcEwen1/BaseAL) is an evaluation framework for active learning methods. Operating at the embedding level, BaseAL is designed to be domain and modality agnostic: any dataset that can be represented as embeddings, such as audio, images, text, and beyond, can be evaluated within the framework. BaseAL simulates oracle labelling using pre-labelled datasets. Importantly, BaseAL ([v1.2](https://github.com/BenMcEwen1/BaseAL/releases)) considers AL within the broader context of budget management, including active testing (validation) mechanisms as well as learning. While active testing has distinct objectives, samples are queried from the same budget. Version 1.2 provides the capacity to evaluate joint allocation and selection policies.

For AL researchers, BaseAL provides an experiment manager for running multiple methods and configurations in parallel with repeated runs, enabling robust evaluation. Standardised evaluation metrics including Area Under the Learning Curve (AULC) are built in, and compatibility with scikit-activeml [@kottke2021scikitactiveml] and other AL libraries gives access to over 40 sampling methods. For domain researchers encountering AL for the first time, BaseAL offers an interactive web application for visualising the embedding space, sample utility, and raw data across AL cycles, alongside minimal-setup notebooks and a simple API. BaseAL has been used as the evaluation infrastructure for the [BioDCASE](https://biodcase.github.io/) [@stowell2026biodcase] *Active Learning for Bioacoustics* data [challenge](https://biodcase.github.io/challenge2026/task4), evaluating AL sampling strategies across both terrestrial and marine bioacoustic datasets.

# Statement of need

Active learning methods are well studied [@settles2012active], but researchers evaluating them face two practical obstacles. First, comparing sampling strategies rigorously requires running many method-and-configuration combinations with repeated trials to average out stochastic variation. Query-strategy libraries such as scikit-activeml [@kottke2021scikitactiveml] and modAL [@danka2018modal] focus on providing the strategies themselves, and dedicated experiment-management tooling for large-scale parallel comparison is available in varying forms across the ecosystem (see State of the field). Second, published comparisons often report only training accuracy against the number of labels queried, while the computational cost of the sampling method itself, the cost of repeated AL cycles, and the cost of annotation are rarely reported alongside performance gains. BaseAL addresses both: it provides an experiment manager for running multiple methods and configurations in parallel with repeated runs, and it reports standardised metrics covering training efficiency, sampling and cycle time, and annotation cost.

For domain researchers outside of machine learning, AL is often unfamiliar despite being directly applicable to a broad range of problem spaces. Label bottlenecks are common wherever only a small proportion of available data can be labelled and labelling requires specialist expertise, yet AL methods are not common knowledge outside the core machine learning community. BaseAL's web [application](https://baseal.up.railway.app/) and embedding-space visualisation are intended to lower this barrier: a user can see model uncertainty, sample selection, and how the embedding space updates across AL cycles, providing insight into how different acquisition functions represent utility without first needing to understand the underlying implementation. This pedagogical framing also suits data challenges; BaseAL was designed with this use case in mind and has already served as the evaluation infrastructure for the BioDCASE Active Learning for Bioacoustics challenge [@stowell2026biodcase].

Both audiences share a need that current AL evaluation platforms do not address: treating validation allocation as a variable rather than fixed. Active testing selects which samples to use for validation and has distinct objectives from AL's training-focused sampling, but in realistic deployments the samples used for each are drawn from the same finite annotation budget. This is a critical consideration once AL is applied outside of benchmark datasets with abundant held-out labels, yet it is overlooked by traditional benchmarks and, to the best of our knowledge, unsupported by existing AL evaluation platforms (see State of the field). BaseAL's `ValidationStrategy` mechanism makes this allocation an explicit, configurable part of the experiment, allowing joint allocation and selection policies to be evaluated directly.

BaseAL is not, however, an annotation platform: evaluation is simulated via oracle labelling of pre-labelled datasets rather than collecting new human annotations. Nor is it a comprehensive library of sampling methods as these are already well served by scikit-activeml, modAL, ALiPy and similar packages [@kottke2021scikitactiveml; @danka2018modal; @tang2019alipy]. Instead, BaseAL imports methods from these libraries as needed, with example usage provided for scikit-activeml.


# State of the field

Existing active learning (AL) software falls into two broad categories relative to BaseAL. The first is query-strategy libraries: scikit-activeml [@kottke2021scikitactiveml], modAL [@danka2018modal], libact [@yang2017libact], small-text [@schroder2021smalltext], BaaL [@atighehchian2020baal], and DeepAL [@huang2021deepal]. These libraries implement and compare sampling strategies which are available within BaseAL.

The second, more closely related category is experiment management. Cardinal [@abraham2022cardinal] and its companion benchmark OpenAL [@jonas2023openal] provide experiment monitoring and interpretability metrics for comparing sampling strategies; ALiPy [@tang2019alipy] additionally supports resumable, cached, multi-run comparison of methods; PyRelationAL [@scherer2022pyrelational] ships curated benchmark datasets and cold/warm-start task configurations; and BaaL [@atighehchian2020baal] includes calibration metrics (Expected Calibration Error) alongside its sampling strategies. ALaaS [@huang2022alaas] and ALToolbox [@tsvigun2022altoolbox] address adjacent MLOps deployment and NLP annotation-tooling use cases, respectively. BaseAL builds on this foundation, contributing an experiment manager for parallel, repeated-run evaluation together with a standardised metric set (AULC, calibration, sampling-time and annotation-cost accounting). Its principal extension, however, is architectural: to our knowledge, none of the tools above allocate the annotation budget itself between AL training queries and active-testing (validation) queries as a jointly-tracked policy; the evaluation designs we reviewed instead assume either a fixed test set [@abraham2022cardinal] or a static held-out validation split [@scherer2022pyrelational]. BaseAL's `ValidationStrategy` extends this line of work by treating validation sampling as an adaptively-selected component of the same shared budget.

BaseAL is also distinct from annotation platforms such as Label Studio [@labelstudio], which support human-in-the-loop labelling workflows (including AL-prioritised labelling) but do not provide sampling-strategy comparison or evaluation infrastructure, and from embedding-visualisation tools such as the Embedding Projector [@smilkov2016embeddingprojector], which visualise embedding spaces without an AL component. Interactive visualisation of model behaviour across AL cycles has been explored as a research prototype [@wang2021interactive]; BaseAL extends this idea into a maintained package integrated with its experiment manager and active-testing mechanism.

# Software design

BaseAL's core is an `ActiveLearner`/`Manager` pair: `ActiveLearner` runs the sample-query-train loop for one configuration (sampling strategy, warm-up strategy, classification model), while `Manager` orchestrates multiple `ActiveLearner` instances in parallel across configurations and repeats. Datasets follow a fixed, modality-agnostic layout with optional raw data, per-model embeddings, and a labels/metadata CSV pair. Any dataset already reduced to embeddings can be evaluated without framework changes:

```
{dataset_name}_BASEAL/
├── data/ (optional)
|   └── {model_name}/
│       ├── {example_1}.wav/.mp3/
│       ├── {example_2}.wav
|       └── ...
├── embeddings/
│   └── {model_name}/
│       ├── {example_1}.npy
│       └── ...
├── labels.csv        # filename, label, validation split
└── metadata.csv      # All metadata
```

## Embedding-level operation

Operating at the embedding level is a deliberate trade-off: it supports the increasingly common *active fine-tuning* setting, where AL is applied over embeddings from a frozen, pre-trained feature extractor, but limits BaseAL to fine-tuning the classification head rather than the full extractor (not supported natively as of v1.2). In return, the same framework evaluates audio, image, or text embeddings unmodified, and avoids redundant model-specific dependencies of feature extraction models. For example, the public [dataset generator](https://github.com/BenMcEwen1/dataset-generator) built on [bacpipe](https://github.com/bioacoustic-ai/bacpipe) was applied for audio embedding. The classification head is fully adjustable in `model.py`, constrained only to match embedding-input and class-output shapes; the default is a lightweight two-layer MLP, with a separate `projection` layer used to visualise how the embedding space evolves across AL cycles.

## Data challenges

BaseAL was designed with data challenges in mind, balancing ease of evaluation against participant freedom. For the BioDCASE Active Learning for Bioacoustics challenge, submissions were constrained to a single `sampling.py` implementing `SamplingStrategy` and `WarmupStrategy`, with full access to embeddings, predictions, metadata, and hyperparameters, while `model.py` and `active_learner.py` were withheld to prevent trivial solutions such as overparameterisation. `ValidationStrategy` (v1.2) mirrors this mechanism for the validation budget.

## Web application

An interactive web [application](https://baseal.up.railway.app/) lets users step through AL cycles, visualising the embedding space, per-sample utility, and (for audio and image datasets) the underlying data, with no local setup required, extending prior visualisation prototypes [@wang2021interactive].

## Evaluation and Reporting

BaseAL exports a YAML file per sampling strategy and dataset, reporting a per-cycle learning curve and summary statistics, with standard deviation reported alongside the mean for every metric across independent repeats.

**Training efficiency** is reported via `mAP` (macro-averaged Average Precision) and its running Area Under the Learning Curve, `aulc_mAP`, BaseAL's headline metric since it summarises the whole label-efficiency trajectory rather than only final performance; accuracy and F1 are tracked identically.

**Computational cost** is reported separately, since a strategy that improves `mAP` at high per-cycle overhead is not necessarily preferable: sampling wall-clock time, annotation cost (labels consumed, which can exceed samples selected under multilabel datasets), and training compute relative to a fixed baseline configuration.

**Validation metrics**, introduced in v1.2 alongside `ValidationStrategy`, give the participant's own running estimate of full-task performance, distinct from both metrics above since this budget is drawn from the same pool as training samples rather than a free held-out set. BaseAL bootstraps the revealed validation subset to produce a per-class performance estimate with an upper and lower confidence bound, then scores that interval with a Winkler (interval) score [@winkler1972decision]. This rewards a narrow interval, but only if it still covers the true performance. Classes without enough revealed labels are treated as unmeasured: rather than left undefined, both the estimate and the truth it is scored against fall back to that class's chance-level prevalence, so a submission is only judged on classes it actually had the opportunity to observe. This interval's running Area Under the Learning Curve summarises performance across cycles.

# Research impact statement

Used in the 2026 BioDCASE Active Learning for Bioacoustics [challenge](https://biodcase.github.io/challenge2026/task4), resulting in 10 unique submissions, from teams across 12 research institutes and companies. Official results (published on the 6th of July, 2026) can be viewed on the BioDCASE results [page](https://biodcase.github.io/challenge2026/task4-results) with participant reports [available](https://biodcase.github.io/challenge2026/task4-results#technical-reports). Results are also available on the [BaseAL](https://baseal.up.railway.app/#/challenges/biodcase) website.

# AI usage disclosure

Generative AI (Claude Code) was used in the development of the BaseAL software package and also in planning phases to determine the overall structure and scope. All AI generated content and design decisions were manually reviewed by authors.

# Acknowledgements

We acknowledge the feedback from the BioDCASE Active Learning for Bioacoustics organising [team](https://biodcase.github.io/challenge2026/#active-learning-for-bioacoustics) and task participants, which helped to identify and solve bugs, improve documentation clarity and add requested features.

# References