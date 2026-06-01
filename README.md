<div style="font-family: charter;">

<table>
<tr>
<td width="82%">
<h1><i>Flat-Pack Bench</i>:<br/>Evaluating Spatio-Temporal Understanding in Large Vision-Language Models through Furniture Assembly</h1>
</td>
<td width="18%" align="right">
<img src="assets/readme/favicon.png" alt="Flat-Pack Bench logo" width="120"/>
</td>
</tr>
</table>

<p>
<a href="https://justachetan.github.io/" target="_blank">Aditya Chetan</a>,
<a href="https://www.linkedin.com/in/ericcai32/" target="_blank">Eric Cai,
<a href="https://peey.github.io/" target="_blank">Peeyush Kushwaha</a>,
<a href="https://bharathrajn.com/" target="_blank">Bharath Raj Nagoor Kani</a>,
<a href="https://utkarshmall.com/" target="_blank">Utkarsh Mall</a>,
<a href="https://qianqianwang68.github.io/" target="_blank">Qianqian Wang</a>,
<a href="https://www.cs.cornell.edu/~snavely/" target="_blank">Noah Snavely</a>,
<a href="https://www.cs.cornell.edu/~bharathh/" target="_blank">Bharath Hariharan</a>
</p>

<p><strong><a href="https://cvpr.thecvf.com/" target="_blank">CVPR 2026</a></strong></p>

<p>
<a href="https://arxiv.org/abs/2605.21625" target="_blank">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-2605.21625-red?logo=arxiv" height="20" />
</a>
<a href="https://flat-pack-bench.github.io/" target="_blank">
    <img alt="Website" src="https://img.shields.io/badge/🌎_Website-Flat--Pack--Bench-blue.svg" height="20" />
</a>
<a href="https://huggingface.co/collections/justachetan/flat-pack-bench" target="_blank">
    <img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20_Benchmark-Flat--Pack--Bench-ffc107?color=ffc107&logoColor=white" height="20" />
</a>
</p>

</div>

## Overview

<p align="justify"><i>The emergence of Large Vision-Language Models (LVLMs) has significantly advanced video understanding capabilities. However, existing benchmarks focus predominantly on coarse-grained tasks such as action segmentation, classification, captioning, and retrieval. Furthermore, these benchmarks often rely on entities that can be easily identified verbally, like household objects, animals, human subjects, etc., limiting their applicability to complex, in-the-wild video scenarios. But, many applications such as furniture assembly, cooking, etc., require step-by-step fine-grained spatio-temporal understanding of the video, which is not sufficiently evaluated in current benchmarks. To address this gap, we introduce Flat-Pack Bench, a novel benchmark centered on furniture assembly tasks. Our benchmark evaluates LVLMs on nuanced tasks, including temporal ordering of assembly actions, temporal localization of assembly state, understanding part mating, and tracking, using multiple-choice questions paired with visual prompts highlighting relevant parts as references for fine-grained questions. Our experiments reveal that state-of-the-art LVLMs struggle significantly with fine-grained spatio-temporal reasoning, highlighting their limitations in effectively leveraging temporal information from videos, limited tracking ability, and understanding of spatial interactions like physical contact.</i></p>

## Setup

Installation and environment setup instructions are maintained in [setup/README.md](setup/README.md). The setup guide covers the default `fpb` environment plus the `fpb-llava`, `fpb-plm`, and `fpb-sam2` environments used by model-specific experiments.

## Experiments

Experiment and evaluation code lives under [src/](src/). See [src/eval/](src/eval/) for benchmark inference, prompt rendering, evaluation, and tabulation, and [src/tva/](src/tva/) for Temporal Video Agent experiments.

## Run Evaluations

To run model evaluations, start with the evaluation package documentation in [src/eval/README.md](src/eval/README.md). It documents the Hydra configs, media pipelines, model wrappers, inference entrypoint, scoring scripts, and result tabulation utilities.

Dataset download and local data layout details are documented in [data/README.md](data/README.md).

## Limitations

While we strive to maintain the highest quality of annotations, some imperfections might exist. If you notice annotation issues, ambiguous questions, or other dataset problems, please point them out to us so we can improve the benchmark.

## Acknowledgements

Flat-Pack Bench builds on [IKEA Manuals at Work](https://yunongliu1.github.io/ikea-video-manual/), whose furniture assembly videos and annotations made this benchmark possible. We thank the IKEA-Manuals-At-Work authors for releasing this valuable resource.

## Citation

If you use Flat-Pack Bench in your research, please consider citing our paper:

```bibtex
@InProceedings{Chetan_2026_CVPR,
    author    = {Chetan, Aditya and Cai, Eric and Kushwaha, Peeyush and Kani, Bharath Raj Nagoor and Mall, Utkarsh and Wang, Qianqian and Snavely, Noah and Hariharan, Bharath},
    title     = {Flat-Pack Bench: Evaluating Spatio-Temporal Understanding in Large Vision-Language Models through Furniture Assembly},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2026},
    pages     = {16624-16634}
}
```
