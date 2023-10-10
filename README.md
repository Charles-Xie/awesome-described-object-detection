# awesome-described-object-detection
A curated list of papers and resources related to Described Object Detection, Open-Vocabulary/Open-World Object Detection and Referring Expression Comprehension.

## Table of Contents

- [Awesome Papers](#awesome-papers)
    - [Described Object Detection](#described-object-detection)
        - [Methods with Potential for DOD](#methods-with-potential-for-dod)
    - [Open-Vocabulary Object Detection](#open-vocabulary-object-detection)
    - [Referring Expression Comprehension](#referring-expression-comprehension)
- [Awesome Datasets](#awesome-datasets)
- [Related Surveys and Resources](#related-surveys-and-resources)

# Awesome Papers

## Described Object Detection

- Exposing the Troublemakers in Described Object Detection (NeurIPS 2023) [[paper]](https://arxiv.org/abs/2307.12813) [[dataset]](https://github.com/shikras/d-cube/) [[code]](https://github.com/shikras/d-cube/)![Star](https://img.shields.io/github/stars/shikras/d-cube.svg?style=social&label=Star)

### Methods with Potential for DOD

These methods are either MLLM with capabilities related to detection/localization, or multi-task models handling both OD/OVD and REC. Though they are not directly handling DOD and not evaluated on DOD benchmarks in their original papers, it is possible that they obtain a performance similar to the DOD baseline.

- Contextual Object Detection with Multimodal Large Language Models (arxiv 2023) [[paper]](https://arxiv.org/abs/2305.18279) [[demo]](https://huggingface.co/spaces/yuhangzang/ContextDet-Demo) [[code]](https://github.com/yuhangzang/ContextDET)![Star](https://img.shields.io/github/stars/yuhangzang/ContextDET.svg?style=social&label=Star)

- Kosmos-2: Grounding Multimodal Large Language Models to the World (arxiv 2023) [[paper]](https://arxiv.org/abs/2306.14824) [[demo]](https://huggingface.co/spaces/ydshieh/Kosmos-2) [[code]](https://github.com/microsoft/unilm/tree/master/kosmos-2)![Star](https://img.shields.io/github/stars/microsoft/unilm.svg?style=social&label=Star)

- Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond (arxiv 2023) [[paper]](https://arxiv.org/abs/2308.12966) [[demo]](https://modelscope.cn/studios/qwen/Qwen-VL-Chat-Demo/summary) [[code]](https://github.com/QwenLM/Qwen-VL)![Star](https://img.shields.io/github/stars/QwenLM/Qwen-VL.svg?style=social&label=Star)

- Shikra: Unleashing Multimodal LLM’s Referential Dialogue Magic (arxiv 2023) [[paper]](https://arxiv.org/abs/2306.15195) [[demo]](http://demo.zhaozhang.net:7860/) [[code]](https://github.com/shikras/shikra)![Star](https://img.shields.io/github/stars/shikras/shikra.svg?style=social&label=Star)

- Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection (arxiv 2023) [[paper]](https://arxiv.org/abs/2303.05499) [[code (eval)]](https://github.com/IDEA-Research/GroundingDINO)![Star](https://img.shields.io/github/stars/IDEA-Research/GroundingDINO.svg?style=social&label=Star) (REC, OD, etc.)

- Universal Instance Perception as Object Discovery and Retrieval (CVPR 2023) [[paper]](https://arxiv.org/abs/2303.06674v2) [[code]](https://github.com/MasterBin-IIAU/UNINEXT)![Star](https://img.shields.io/github/stars/MasterBin-IIAU/UNINEXT.svg?style=social&label=Star) (REC, OVD, etc.)

- Coarse-to-Fine Vision-Language Pre-training with Fusion in the Backbone (NeurIPS 2022) [[paper]](https://arxiv.org/abs/2206.07643) [[code]](https://github.com/microsoft/FIBER)![Star](https://img.shields.io/github/stars/microsoft/FIBER.svg?style=social&label=Star)
- FindIt: Generalized Localization with Natural Language Queries (ECCV 2022) [[paper]](https://arxiv.org/abs/2203.17273) [[code]](https://github.com/google-research/google-research/tree/master/findit)![Star](https://img.shields.io/github/stars/google-research/google-research.svg?style=social&label=Star) (REC, OD, etc.)

- GRiT: A Generative Region-to-text Transformer for Object Understanding (arxiv 2022) [[paper]](https://arxiv.org/abs/2212.00280) [[demo (colab)]](https://colab.research.google.com/github/taskswithcode/GriT/blob/master/TWCGRiT.ipynb) [[code]](https://github.com/JialianW/GRiT)![Star](https://img.shields.io/github/stars/JialianW/GRiT.svg?style=social&label=Star)


## Open-Vocabulary Object Detection

- DST-Det: Simple Dynamic Self-Training for Open-Vocabulary Object Detection (arxiv 2023) [[paper]](https://arxiv.org/abs/2310.01393) [[code (soon)]](https://github.com/xushilin1/dst-det)

- Detection-Oriented Image-Text Pretraining for Open-Vocabulary Detection (arxiv 2023) [[paper]](https://arxiv.org/abs/2310.00161)

- Exploring Multi-Modal Contextual Knowledge for Open-Vocabulary Object Detection (arxiv 2023) [[paper]](https://arxiv.org/abs/2308.15846)

- How to Evaluate the Generalization of Detection? A Benchmark for Comprehensive Open-Vocabulary Detection (arxiv 2023) [[paper]](https://arxiv.org/abs/2308.13177) [[dataset]](https://github.com/om-ai-lab/OVDEval)

- Improving Pseudo Labels for Open-Vocabulary Object Detection (arxiv 2023) [[paper]](https://arxiv.org/abs/2308.06412)

- Scaling Open-Vocabulary Object Detection (arxiv 2023) [[paper]](https://arxiv.org/abs/2306.09683) [[code (jax)]](https://github.com/google-research/scenic/tree/main/scenic/projects/owl_vit)

- Unified Open-Vocabulary Dense Visual Prediction (arxiv 2023) [[paper]](https://arxiv.org/abs/2307.08238)

- Open-Vocabulary Object Detection via Scene Graph Discovery (arxiv 2023) [[paper]](https://arxiv.org/abs/2307.03339)

- Three Ways to Improve Feature Alignment for Open Vocabulary Detection (arXiv 2023) [[paper]](https://arxiv.org/abs/2303.13518)

- Prompt-Guided Transformers for End-to-End Open-Vocabulary Object Detection (arXiv 2023) [[paper]](https://arxiv.org/abs/2303.14386)

- Open-Vocabulary Object Detection using Pseudo Caption Labels (arXiv 2023) [[paper]](https://arxiv.org/abs/2303.13040)

- What Makes Good Open-Vocabulary Detector: A Disassembling Perspective (KDD Workshop) [[paper]](https://arxiv.org/abs/2309.00227)

- Open-Vocabulary Object Detection With an Open Corpus (ICCV 2023) [[paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Wang_Open-Vocabulary_Object_Detection_With_an_Open_Corpus_ICCV_2023_paper.pdf)

- Distilling DETR with Visual-Linguistic Knowledge for Open-Vocabulary Object Detection (ICCV 2023) [[paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Li_Distilling_DETR_with_Visual-Linguistic_Knowledge_for_Open-Vocabulary_Object_Detection_ICCV_2023_paper.pdf) [[code (soon)]](https://github.com/hikvision-research/opera)

- A Simple Framework for Open-Vocabulary Segmentation and Detection (ICCV 2023) [[paper]](https://arxiv.org/abs/2303.08131) [[code]](https://github.com/IDEA-Research/OpenSeeD)

- EdaDet: Open-Vocabulary Object Detection Using Early Dense Alignment (ICCV 2023) [[paper]](https://arxiv.org/abs/2309.01151) [[website]](https://chengshiest.github.io/edadet/)

- Contrastive Feature Masking Open-Vocabulary Vision Transformer (ICCV 2023) [[paper]](https://arxiv.org/abs/2309.00775)

- Multi-Modal Classifiers for Open-Vocabulary Object Detection (ICML 2023) [[paper]](http://arxiv.org/abs/2306.05493) [[code (eval)]](https://github.com/prannaykaul/mm-ovod)

- CORA: Adapting CLIP for Open-Vocabulary Detection with Region Prompting and Anchor Pre-Matching (CVPR 2023) [[paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Wu_CORA_Adapting_CLIP_for_Open-Vocabulary_Detection_With_Region_Prompting_and_CVPR_2023_paper.pdf) [[code]](https://github.com/tgxs002/CORA)

- Object-Aware Distillation Pyramid for Open-Vocabulary Object Detection (CVPR 2023) [[paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_Object-Aware_Distillation_Pyramid_for_Open-Vocabulary_Object_Detection_CVPR_2023_paper.pdf) [[code]](https://github.com/LutingWang/OADP)

- Aligning Bag of Regions for Open-Vocabulary Object Detection (CVPR 2023) [[paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Wu_Aligning_Bag_of_Regions_for_Open-Vocabulary_Object_Detection_CVPR_2023_paper.pdf) [[code]](https://github.com/wusize/ovdet)

- Region-Aware Pretraining for Open-Vocabulary Object Detection with Vision Transformers (CVPR 2023) [[paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Kim_Region-Aware_Pretraining_for_Open-Vocabulary_Object_Detection_With_Vision_Transformers_CVPR_2023_paper.pdf) [[code]](https://github.com/google-research/google-research/tree/master/fvlm/rovit)

- DetCLIPv2: Scalable Open-Vocabulary Object Detection Pre-training via Word-Region Alignment (CVPR 2023) [[paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Yao_DetCLIPv2_Scalable_Open-Vocabulary_Object_Detection_Pre-Training_via_Word-Region_Alignment_CVPR_2023_paper.pdf)

- Learning to Detect and Segment for Open Vocabulary Object Detection (CVPR 2023) [[paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_Learning_To_Detect_and_Segment_for_Open_Vocabulary_Object_Detection_CVPR_2023_paper.pdf)

- F-VLM: Open-Vocabulary Object Detection upon Frozen Vision and Language Models (ICLR 2023) [[paper]](https://openreview.net/pdf?id=MIMwy4kh9lf) [[code]] (https://github.com/google-research/google-research/tree/master/fvlm) [[website]](https://sites.google.com/view/f-vlm/home)

- Learning Object-Language Alignments for Open-Vocabulary Object Detection (ICLR 2023) [[paper]](https://openreview.net/pdf?id=mjHlitXvReu) [[code]](https://github.com/clin1223/VLDet)

- Simple Open-Vocabulary Object Detection with Vision Transformers (ECCV 2022) [[paper]](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136700714.pdf) [[code (jax)]](https://github.com/google-research/scenic/tree/main/scenic/projects/owl_vit) [[code (huggingface)]](https://huggingface.co/docs/transformers/model_doc/owlvit)

- P<sup>3</sup>OVD: Fine-grained Visual-Text Prompt-Driven Self-Training for Open-Vocabulary Object Detection (arXiv 2022) [[paper]](https://arxiv.org/abs/2211.00849)

- Open Vocabulary Object Detection with Proposal Mining and Prediction Equalization (arXiv 2022) [[paper]](https://arxiv.org/abs/2206.11134) [[code]](https://github.com/PeixianChen/MEDet)

- Localized Vision-Language Matching for Open-vocabulary Object Detection (GCPR 2022) [[paper]](https://arxiv.org/abs/2205.06160) [[code]](https://github.com/lmb-freiburg/locov)

- Bridging the Gap between Object and Image-level Representations for Open-Vocabulary Detection (NeurIPS 2022) [[paper]](https://openreview.net/forum?id=aKXBrj0DHm) [[code]](https://github.com/hanoonaR/object-centric-ovd)

- X-DETR: A Versatile Architecture for Instance-wise Vision-Language Tasks (ECCV 2022) [[paper]](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136960288.pdf)

- Exploiting Unlabeled Data with Vision and Language Models for Object Detection (ECCV 2022) [[paper]](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136690156.pdf) [[code]](https://github.com/xiaofeng94/VL-PLM)

- PromptDet: Towards Open-vocabulary Detection using Uncurated Images (ECCV 2022) [[paper]](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136690691.pdf) [[code]](https://github.com/fcjian/PromptDet)

- Open-Vocabulary DETR with Conditional Matching (ECCV 2022) [[paper]](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136690107.pdf) [[code]](https://github.com/yuhangzang/OV-DETR)

- Open Vocabulary Object Detection with Pseudo Bounding-Box Labels (ECCV 2022) [[paper]](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136700263.pdf) [[code]](https://github.com/salesforce/PB-OVD)

- Simple Open-Vocabulary Object Detection with Vision Transformers (ECCV 2022) [[paper]](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136700714.pdf) [[code]](https://github.com/google-research/scenic/tree/main/scenic/projects/owl_vit)

- RegionCLIP: Region-Based Language-Image Pretraining (CVPR 2022) [[paper]](https://openaccess.thecvf.com/content/CVPR2022/html/Zhong_RegionCLIP_Region-Based_Language-Image_Pretraining_CVPR_2022_paper.html) [[code]](https://github.com/microsoft/RegionCLIP)

- Open-Vocabulary Instance Segmentation via Robust Cross-Modal Pseudo-Labeling (CVPR 2022) [[paper]](https://openaccess.thecvf.com/content/CVPR2022/html/Huynh_Open-Vocabulary_Instance_Segmentation_via_Robust_Cross-Modal_Pseudo-Labeling_CVPR_2022_paper.html) [[code]](https://github.com/hbdat/cvpr22_cross_modal_pseudo_labeling)

- Open-Vocabulary One-Stage Detection With Hierarchical Visual-Language Knowledge Distillation (CVPR 2022) [[paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Ma_Open-Vocabulary_One-Stage_Detection_With_Hierarchical_Visual-Language_Knowledge_Distillation_CVPR_2022_paper.pdf) [[code]](https://github.com/mengqiDyangge/HierKD)

- Learning to Prompt for Open-Vocabulary Object Detection with Vision-Language Model (CVPR 2022) [[paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Du_Learning_To_Prompt_for_Open-Vocabulary_Object_Detection_With_Vision-Language_Model_CVPR_2022_paper.pdf) [[code]](https://github.com/dyabel/detpro)

- Open-vocabulary Object Detection via Vision and Language Knowledge Distillation (ICLR 2022) [[paper]](https://openreview.net/forum?id=lL3lnMbR4WU) [[code]](https://github.com/tensorflow/tpu/tree/master/models/official/detection/projects/vild)

- Open-Vocabulary Object Detection Using Captions (CVPR 2021) [[paper]](https://openaccess.thecvf.com/content/CVPR2021/papers/Zareian_Open-Vocabulary_Object_Detection_Using_Captions_CVPR_2021_paper.pdf) [[code]](https://github.com/alirezazareian/ovr-cnn)

## Referring Expression Comprehension

- ONE-PEACE: Exploring One General Representation Model Toward Unlimited Modalities [[paper]](https://arxiv.org/abs/2305.11172) [[code]](https://github.com/OFA-Sys/ONE-PEACE)

- Unified-IO: A Unified Model for Vision, Language, and Multi-Modal Tasks (ICLR 2023) [[paper]](https://arxiv.org/abs/2206.08916) [[code (eval)]](https://github.com/allenai/unified-io-inference)

- OFA: Unifying Architectures, Tasks, and Modalities Through a Simple Sequence-to-Sequence Learning Framework (ICML 2022) [[paper]](https://arxiv.org/abs/2202.03052) [[code]](https://github.com/OFA-Sys/OFA)

- PolyFormer: Referring Image Segmentation as Sequential Polygon Generation (CVPR 2023) [[paper]](https://arxiv.org/abs/2302.07387) [[website]](https://polyformer.github.io/) [[code]](https://github.com/amazon-science/polygon-transformer) [[demo]](https://huggingface.co/spaces/koajoel/PolyFormer)

- DQ-DETR: Dual Query Detection Transformer for Phrase Extraction and Grounding (AAAI 2023) [[paper]](https://arxiv.org/abs/2211.15516) [[code]](https://github.com/IDEA-Research/DQ-DETR)

- Correspondence Matters for Video Referring Expression Comprehension (https://dl.acm.org/doi/abs/10.1145/3503161.3547756) [[paper]](https://dl.acm.org/doi/abs/10.1145/3503161.3547756)

- MDETR -- Modulated Detection for End-to-End Multi-Modal Understanding (ICCV 2021) [[paper]](https://arxiv.org/abs/2104.12763) [[website]](https://ashkamath.github.io/mdetr_page/) [[code]](https://github.com/ashkamath/mdetr)![Star](https://img.shields.io/github/stars/ashkamath/mdetr.svg?style=social&label=Star)

- Referring Transformer: A One-step Approach to Multi-task Visual Grounding (NeurIPS 2021) [[paper]](https://arxiv.org/abs/2106.03089) [[code]](https://github.com/ubc-vision/RefTR)

- Large-Scale Adversarial Training for Vision-and-Language Representation Learning (NIPS 2023 Spotlight) [[paper]](https://arxiv.org/abs/2006.06195) [[code]](https://github.com/zhegan27/VILLA) [[poster]](https://zhegan27.github.io/Papers/villa_poster.pdf)

- UNITER: UNiversal Image-TExt Representation Learning (ECCV 2020) [[paper]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123750103.pdf) [[code]](https://github.com/ChenRocks/UNITER)

- Multi-task Collaborative Network for Joint Referring Expression Comprehension and Segmentation (CVPR 2020) [[paper]](https://arxiv.org/abs/2003.08813) [[code]](https://github.com/luogen1996/MCN)

- A Real-Time Cross-modality Correlation Filtering Method for Referring
Expression Comprehension (CVPR 2020) [[paper]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Liao_A_Real-Time_Cross-Modality_Correlation_Filtering_Method_for_Referring_Expression_Comprehension_CVPR_2020_paper.pdf)

- Dynamic Graph Attention for Referring Expression Comprehension (ICCV 2019) [[paper]](https://openaccess.thecvf.com/content_ICCV_2019/html/Yang_Dynamic_Graph_Attention_for_Referring_Expression_Comprehension_ICCV_2019_paper.html)

- Neighbourhood Watch: Referring Expression Comprehension via Language-Guided Graph Attention Networks (CVPR 2019) [[paper]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Neighbourhood_Watch_Referring_Expression_Comprehension_via_Language-Guided_Graph_Attention_Networks_CVPR_2019_paper.pdf)

- MAttNet: Modular Attention Network for Referring Expression Comprehension (CVPR 2018) [[paper]](https://arxiv.org/abs/1801.08186) [[code]](https://github.com/lichengunc/MAttNet)

- Comprehension-Guided Referring Expressions (CVPR 2017) [[paper]](https://openaccess.thecvf.com/content_cvpr_2017/papers/Luo_Comprehension-Guided_Referring_Expressions_CVPR_2017_paper.pdf)

- Modeling Context Between Objects for Referring Expression Understanding (ECCV 2016) [[paper]](https://link.springer.com/chapter/10.1007/978-3-319-46493-0_48)


# Awesome Datasets

## Datasets for DOD and Similar Tasks

## Detection Datasets

## Grounding Datasets

| Name | Paper | Website | Code | Train/Eval | Notes |
|:-----|:-----:|:----:|:-----:|:-----:|:-----:|
| **GRiT** | [GRIT: General Robust Image Task Benchmark](https://arxiv.org/abs/2204.13653) | [Link](https://allenai.org/project/grit/home) | [Github](https://github.com/allenai/grit_official) | eval only |  |



# Related Surveys and Resources

Some survey papers regarding relevant tasks (open-vocabulary learning, etc.)

- Towards Open Vocabulary Learning: A Survey (arxiv 2023) [[paper]](https://arxiv.org/abs/2306.15880) [[repo]](https://github.com/jianzongwu/Awesome-Open-Vocabulary)
- A Survey on Open-Vocabulary Detection and Segmentation: Past, Present, and Future (arxiv 2023) [[paper]](https://arxiv.org/abs/2307.09220)

Some similar github repos like awesome lists:

- [daqingliu/awesome-rec](https://github.com/daqingliu/awesome-rec): A curated list of REC papers. Not maintained in recent years.
- [qy-feng/awesome-visual-grounding](https://github.com/qy-feng/awesome-visual-grounding): A curated list of visual grounding papers. Not maintained in recent years.
- [MarkMoHR/Awesome-Referring-Image-Segmentation](https://github.com/MarkMoHR/Awesome-Referring-Image-Segmentation): A list of Referring Expression Segmentation (RES) papers and resources.
- [TheShadow29/awesome-grounding](https://github.com/TheShadow29/awesome-grounding): A list of visual grounding (REC) paper roadmaps and datasets.
- [witnessai/Awesome-Open-Vocabulary-Object-Detection](https://github.com/witnessai/Awesome-Open-Vocabulary-Object-Detection/blob/main/README.md?plain=1): A list of Open-Vocabulary Object Detection papers.


---
<br> **📑 If you find our projects helpful to your research, please consider citing:** <br>

```bibtex
@inproceedings{xie2023DOD,
  title={Exposing the Troublemakers in Described Object Detection},
  author={Xie, Chi and Zhang, Zhao and Wu, Yixuan and Zhu, Feng and Zhao, Rui and Liang, Shuang},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems (NeurIPS)},
  year={2023}
}
```

---

# Acknowledgement

The structure and format of this repo is inspired by [BradyFU/Awesome-Multimodal-Large-Language-Models](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models).
