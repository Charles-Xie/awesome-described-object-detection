[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)
[![PR's Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat)](https://github.com/Charles-Xie/awesome-described-object-detection)

# Awesome Described Object Detection

A curated list of papers and resources related to [Described Object Detection](https://arxiv.org/abs/2307.12813), Open-Vocabulary/Open-World Object Detection and Referring Expression Comprehension.

If you find any work or resources missing, please send a [pull requests](https://github.com/Charles-Xie/awesome-described-object-detection/pulls). Thanks!

---
<br> **📑 If you find our projects helpful to your research, please consider citing:** <br>

```bibtex
@inproceedings{xie2023DOD,
  title={Described Object Detection: Liberating Object Detection with Flexible Expressions},
  author={Xie, Chi and Zhang, Zhao and Wu, Yixuan and Zhu, Feng and Zhao, Rui and Liang, Shuang},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems (NeurIPS)},
  year={2023}
}
```

---


## Table of Contents

- [Awesome Papers](#awesome-papers)
    - [Described Object Detection](#described-object-detection)
        - [Methods with Potential for DOD](#methods-with-potential-for-dod)
    - [Open-Vocabulary Object Detection](#open-vocabulary-object-detection)
    - [Referring Expression Comprehension/Visual Grounding](#referring-expression-comprehensionvisual-grounding)
- [Awesome Datasets](#awesome-datasets)
    - [Datasets for DOD and Similar Tasks](#datasets-for-dod-and-similar-tasks)
    - [Detection datasets](#detection-datasets)
    - [Grounding Datasets](#grounding-datasets)
- [Related Surveys and Resources](#related-surveys-and-resources)

# Awesome Papers

## Described Object Detection

A leaderboard for update-to-date DOD methods are available [here](https://github.com/shikras/d-cube/tree/main/eval_sota).

- An Open and Comprehensive Pipeline for Unified Object Grounding and Detection (arxiv 2024) [[paper]](https://arxiv.org/abs/2401.02361) [[code]](https://github.com/open-mmlab/mmdetection/tree/main/configs/mm_grounding_dino)![Star](https://img.shields.io/github/stars/open-mmlab/mmdetection.svg?style=social&label=Star)

- Generating Enhanced Negatives for Training Language-Based Object Detectors (arxiv 2024) [[paper]](https://arxiv.org/abs/2401.00094)

- Aligning and Prompting Everything All at Once for Universal Visual Perception (arxiv 2023) [[paper]](https://arxiv.org/abs/2312.02153) [[code]](https://github.com/shenyunhang/APE)![Star](https://img.shields.io/github/stars/shenyunhang/APE.svg?style=social&label=Star)

- Described Object Detection: Liberating Object Detection with Flexible Expressions (NeurIPS 2023) [[paper]](https://arxiv.org/abs/2307.12813) [[dataset]](https://github.com/shikras/d-cube/) [[code]](https://github.com/shikras/d-cube/)![Star](https://img.shields.io/github/stars/shikras/d-cube.svg?style=social&label=Star)


### Methods with Potential for DOD

These methods are either MLLM with capabilities related to detection/localization, or multi-task models handling both OD/OVD and REC. Though they are not directly handling DOD and not evaluated on DOD benchmarks in their original papers, it is possible that they obtain a performance similar to the DOD baseline.

- LLMs Meet VLMs: Boost Open Vocabulary Object Detection with Fine-grained Descriptors (ICLR 2024) [[paper]](https://arxiv.org/abs/2402.04630)

- SPHINX: The Joint Mixing of Weights, Tasks, and Visual Embeddings for Multi-modal Large Language Models (arxiv 2023) [[paper]](https://arxiv.org/abs/2311.07575) [[code]](https://github.com/Alpha-VLLM/LLaMA2-Accessory)![Star](https://img.shields.io/github/stars/Alpha-VLLM/LLaMA2-Accessory.svg?style=social&label=Star)

- Pink: Unveiling the Power of Referential Comprehension for Multi-modal LLMs (arxiv 2023) [[paper]](https://arxiv.org/abs/2310.00582) [[code (soon)]](https://github.com/SY-Xuan/Pink)![Star](https://img.shields.io/github/stars/SY-Xuan/Pink.svg?style=social&label=Star)

- Griffon: Spelling out All Object Locations at Any Granularity with Large Language Models (arxiv 2023) [[paper]](https://arxiv.org/abs/2311.14552) [[code]](https://github.com/jefferyZhan/Griffon)![Star](https://img.shields.io/github/stars/jefferyZhan/Griffon.svg?style=social&label=Star)

- Ferret: Refer and Ground Anything Anywhere at Any Granularity [[paper]](https://arxiv.org/abs/2310.07704) [[code]](https://github.com/apple/ml-ferret)![Star](https://img.shields.io/github/stars/apple/ml-ferret.svg?style=social&label=Star)

- Contextual Object Detection with Multimodal Large Language Models (arxiv 2023) [[paper]](https://arxiv.org/abs/2305.18279) [[demo]](https://huggingface.co/spaces/yuhangzang/ContextDet-Demo) [[code]](https://github.com/yuhangzang/ContextDET)![Star](https://img.shields.io/github/stars/yuhangzang/ContextDET.svg?style=social&label=Star)

- Kosmos-2: Grounding Multimodal Large Language Models to the World (ICLR 2024) [[paper]](https://arxiv.org/abs/2306.14824) [[demo]](https://huggingface.co/spaces/ydshieh/Kosmos-2) [[code]](https://github.com/microsoft/unilm/tree/master/kosmos-2)![Star](https://img.shields.io/github/stars/microsoft/unilm.svg?style=social&label=Star)

- Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond (arxiv 2023) [[paper]](https://arxiv.org/abs/2308.12966) [[demo]](https://modelscope.cn/studios/qwen/Qwen-VL-Chat-Demo/summary) [[code]](https://github.com/QwenLM/Qwen-VL)![Star](https://img.shields.io/github/stars/QwenLM/Qwen-VL.svg?style=social&label=Star)

- Shikra: Unleashing Multimodal LLM’s Referential Dialogue Magic (arxiv 2023) [[paper]](https://arxiv.org/abs/2306.15195) [[demo]](http://demo.zhaozhang.net:7860/) [[code]](https://github.com/shikras/shikra)![Star](https://img.shields.io/github/stars/shikras/shikra.svg?style=social&label=Star)

- Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection (arxiv 2023) [[paper]](https://arxiv.org/abs/2303.05499) [[code (eval)]](https://github.com/IDEA-Research/GroundingDINO)![Star](https://img.shields.io/github/stars/IDEA-Research/GroundingDINO.svg?style=social&label=Star) (REC, OD, etc.)

- Universal Instance Perception as Object Discovery and Retrieval (CVPR 2023) [[paper]](https://arxiv.org/abs/2303.06674v2) [[code]](https://github.com/MasterBin-IIAU/UNINEXT)![Star](https://img.shields.io/github/stars/MasterBin-IIAU/UNINEXT.svg?style=social&label=Star) (REC, OVD, etc.)

- Coarse-to-Fine Vision-Language Pre-training with Fusion in the Backbone (NeurIPS 2022) [[paper]](https://arxiv.org/abs/2206.07643) [[code]](https://github.com/microsoft/FIBER)![Star](https://img.shields.io/github/stars/microsoft/FIBER.svg?style=social&label=Star)
- FindIt: Generalized Localization with Natural Language Queries (ECCV 2022) [[paper]](https://arxiv.org/abs/2203.17273) [[code]](https://github.com/google-research/google-research/tree/master/findit)![Star](https://img.shields.io/github/stars/google-research/google-research.svg?style=social&label=Star) (REC, OD, etc.)

- GRiT: A Generative Region-to-text Transformer for Object Understanding (arxiv 2022) [[paper]](https://arxiv.org/abs/2212.00280) [[demo (colab)]](https://colab.research.google.com/github/taskswithcode/GriT/blob/master/TWCGRiT.ipynb) [[code]](https://github.com/JialianW/GRiT)![Star](https://img.shields.io/github/stars/JialianW/GRiT.svg?style=social&label=Star)


## Open-Vocabulary Object Detection

- InstaGen: Enhancing Object Detection by Training on Synthetic Dataset (arxiv 2024) [[paper]](https://arxiv.org/abs/2402.05937)

- YOLO-World: Real-Time Open-Vocabulary Object Detection (arxiv 2024) [[paper]](https://arxiv.org/abs/2401.17270) [[code]](https://github.com/AILab-CVC/YOLO-World)

- CLIM: Contrastive Language-Image Mosaic for Region Representation (AAAI 2024) [[paper]](https://arxiv.org/abs/2312.11376) [[code]](https://github.com/wusize/CLIM)
 
- Simple Image-level Classification Improves Open-vocabulary Object Detection (arxiv 2023) [[paper]](https://arxiv.org/abs/2312.10439) [[code]](https://github.com/mala-lab/SIC-CADS)

- ProxyDet: Synthesizing Proxy Novel Classes via Classwise Mixup for Open Vocabulary Object Detection (AAAI 2024) [[paper]](https://arxiv.org/abs/2312.07266)

- OpenSD: Unified Open-Vocabulary Segmentation and Detection (arxiv 2023) [[paper]](https://arxiv.org/abs/2312.06703) [[code (soon)]](https://github.com/strongwolf/OpenSD)

- Boosting Segment Anything Model Towards Open-Vocabulary Learning (arxiv 2023) [[paper]](https://arxiv.org/abs/2312.03628)

- Learning Pseudo-Labeler beyond Noun Concepts for Open-Vocabulary Object Detection (arxiv 2023) [[paper]](https://arxiv.org/abs/2312.02103)

- Language-conditioned Detection Transformer (arxiv 2023) [[paper]](https://arxiv.org/abs/2311.17902) [[code]](https://github.com/janghyuncho/DECOLA)

- The devil is in the fine-grained details: Evaluating open-vocabulary object detectors for fine-grained understanding (arxiv 2023) [[paper]](https://arxiv.org/abs/2311.17518) [[code]](https://github.com/lorebianchi98/FG-OVD)

- LP-OVOD: Open-Vocabulary Object Detection by Linear Probing (WACV 2024) [[paper]](https://arxiv.org/abs/2310.17109) [[code (soon)]](https://github.com/VinAIResearch/LP-OVOD)

- Meta-Adapter: An Online Few-shot Learner for Vision-Language Model (NeurIPS 2023) [[paper]](https://arxiv.org/abs/2311.03774)

- Open-Vocabulary Object Detection with Meta Prompt Representation and Instance Contrastive Optimization (BMVC 2023) [[paper]](https://papers.bmvc2023.org/0093.pdf)

- CoDet: Co-Occurrence Guided Region-Word Alignment for Open-Vocabulary Object Detection (NeurIPS 2023) [[paper]](https://arxiv.org/abs/2310.16667) [[code]](https://github.com/CVMI-Lab/CoDet)

- DST-Det: Simple Dynamic Self-Training for Open-Vocabulary Object Detection (arxiv 2023) [[paper]](https://arxiv.org/abs/2310.01393) [[code (soon)]](https://github.com/xushilin1/dst-det)

- Detection-Oriented Image-Text Pretraining for Open-Vocabulary Detection (arxiv 2023) [[paper]](https://arxiv.org/abs/2310.00161)

- Exploring Multi-Modal Contextual Knowledge for Open-Vocabulary Object Detection (arxiv 2023) [[paper]](https://arxiv.org/abs/2308.15846)

- How to Evaluate the Generalization of Detection? A Benchmark for Comprehensive Open-Vocabulary Detection (arxiv 2023) [[paper]](https://arxiv.org/abs/2308.13177) [[dataset]](https://github.com/om-ai-lab/OVDEval)

- Improving Pseudo Labels for Open-Vocabulary Object Detection (arxiv 2023) [[paper]](https://arxiv.org/abs/2308.06412)

- Scaling Open-Vocabulary Object Detection (arxiv 2023) [[paper]](https://arxiv.org/abs/2306.09683) [[code (jax)]](https://github.com/google-research/scenic/tree/main/scenic/projects/owl_vit)

- Unified Open-Vocabulary Dense Visual Prediction (arxiv 2023) [[paper]](https://arxiv.org/abs/2307.08238)

- TIB: Detecting Unknown Objects Via Two-Stream Information Bottleneck (TPAMI 2023) [[paper]](https://ieeexplore.ieee.org/abstract/document/10275124)

- Fine-grained Visual-Text Prompt-Driven Self-Training for Open-Vocabulary Object Detection (TNNLS 2023) [[paper]](https://ieeexplore.ieee.org/abstract/document/10197240)

- Open-Vocabulary Object Detection via Scene Graph Discovery (ACM MM 2023) [[paper]](https://arxiv.org/abs/2307.03339)

- Three Ways to Improve Feature Alignment for Open Vocabulary Detection (arXiv 2023) [[paper]](https://arxiv.org/abs/2303.13518)

- Prompt-Guided Transformers for End-to-End Open-Vocabulary Object Detection (arXiv 2023) [[paper]](https://arxiv.org/abs/2303.14386)

- Open-Vocabulary Object Detection using Pseudo Caption Labels (arXiv 2023) [[paper]](https://arxiv.org/abs/2303.13040)

- What Makes Good Open-Vocabulary Detector: A Disassembling Perspective (KDD 2023 Workshop) [[paper]](https://arxiv.org/abs/2309.00227)

- Open-Vocabulary Object Detection With an Open Corpus (ICCV 2023) [[paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Wang_Open-Vocabulary_Object_Detection_With_an_Open_Corpus_ICCV_2023_paper.pdf)

- Distilling DETR with Visual-Linguistic Knowledge for Open-Vocabulary Object Detection (ICCV 2023) [[paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Li_Distilling_DETR_with_Visual-Linguistic_Knowledge_for_Open-Vocabulary_Object_Detection_ICCV_2023_paper.pdf) [[code]](https://github.com/hikvision-research/opera/tree/main/configs/dk-detr)

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

- F-VLM: Open-Vocabulary Object Detection upon Frozen Vision and Language Models (ICLR 2023) [[paper]](https://openreview.net/pdf?id=MIMwy4kh9lf) [[code]](https://github.com/google-research/google-research/tree/master/fvlm) [[website]](https://sites.google.com/view/f-vlm/home)

- Learning Object-Language Alignments for Open-Vocabulary Object Detection (ICLR 2023) [[paper]](https://openreview.net/pdf?id=mjHlitXvReu) [[code]](https://github.com/clin1223/VLDet)

- Simple Open-Vocabulary Object Detection with Vision Transformers (ECCV 2022) [[paper]](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136700714.pdf) [[code (jax)]](https://github.com/google-research/scenic/tree/main/scenic/projects/owl_vit) [[code (huggingface)]](https://huggingface.co/docs/transformers/model_doc/owlvit)

- Open Vocabulary Object Detection with Proposal Mining and Prediction Equalization (arXiv 2022) [[paper]](https://arxiv.org/abs/2206.11134) [[code]](https://github.com/PeixianChen/MEDet)

- Localized Vision-Language Matching for Open-vocabulary Object Detection (GCPR 2022) [[paper]](https://arxiv.org/abs/2205.06160) [[code]](https://github.com/lmb-freiburg/locov)

- Bridging the Gap between Object and Image-level Representations for Open-Vocabulary Detection (NeurIPS 2022) [[paper]](https://openreview.net/forum?id=aKXBrj0DHm) [[code]](https://github.com/hanoonaR/object-centric-ovd)

- X-DETR: A Versatile Architecture for Instance-wise Vision-Language Tasks (ECCV 2022) [[paper]](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136960288.pdf)

- Exploiting Unlabeled Data with Vision and Language Models for Object Detection (ECCV 2022) [[paper]](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136690156.pdf) [[code]](https://github.com/xiaofeng94/VL-PLM)

- PromptDet: Towards Open-vocabulary Detection using Uncurated Images (ECCV 2022) [[paper]](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136690691.pdf) [[website]](https://fcjian.github.io/promptdet/) [[code]](https://github.com/fcjian/PromptDet)

- Open-Vocabulary DETR with Conditional Matching (ECCV 2022) [[paper]](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136690107.pdf) [[code]](https://github.com/yuhangzang/OV-DETR)

- Open Vocabulary Object Detection with Pseudo Bounding-Box Labels (ECCV 2022) [[paper]](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136700263.pdf) [[code]](https://github.com/salesforce/PB-OVD)

- Simple Open-Vocabulary Object Detection with Vision Transformers (ECCV 2022) [[paper]](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136700714.pdf) [[code]](https://github.com/google-research/scenic/tree/main/scenic/projects/owl_vit)

- RegionCLIP: Region-Based Language-Image Pretraining (CVPR 2022) [[paper]](https://openaccess.thecvf.com/content/CVPR2022/html/Zhong_RegionCLIP_Region-Based_Language-Image_Pretraining_CVPR_2022_paper.html) [[code]](https://github.com/microsoft/RegionCLIP)

- Open-Vocabulary Instance Segmentation via Robust Cross-Modal Pseudo-Labeling (CVPR 2022) [[paper]](https://openaccess.thecvf.com/content/CVPR2022/html/Huynh_Open-Vocabulary_Instance_Segmentation_via_Robust_Cross-Modal_Pseudo-Labeling_CVPR_2022_paper.html) [[code]](https://github.com/hbdat/cvpr22_cross_modal_pseudo_labeling)

- Open-Vocabulary One-Stage Detection With Hierarchical Visual-Language Knowledge Distillation (CVPR 2022) [[paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Ma_Open-Vocabulary_One-Stage_Detection_With_Hierarchical_Visual-Language_Knowledge_Distillation_CVPR_2022_paper.pdf) [[code]](https://github.com/mengqiDyangge/HierKD)

- Learning to Prompt for Open-Vocabulary Object Detection with Vision-Language Model (CVPR 2022) [[paper]](https://openaccess.thecvf.com/content/CVPR2022/papers/Du_Learning_To_Prompt_for_Open-Vocabulary_Object_Detection_With_Vision-Language_Model_CVPR_2022_paper.pdf) [[code]](https://github.com/dyabel/detpro)

- Open-vocabulary Object Detection via Vision and Language Knowledge Distillation (ICLR 2022) [[paper]](https://openreview.net/forum?id=lL3lnMbR4WU) [[code]](https://github.com/tensorflow/tpu/tree/master/models/official/detection/projects/vild)

- Open-Vocabulary Object Detection Using Captions (CVPR 2021) [[paper]](https://openaccess.thecvf.com/content/CVPR2021/papers/Zareian_Open-Vocabulary_Object_Detection_Using_Captions_CVPR_2021_paper.pdf) [[code]](https://github.com/alirezazareian/ovr-cnn)


## Referring Expression Comprehension/Visual Grounding

- GroundVLP: Harnessing Zero-shot Visual Grounding from Vision-Language Pre-training and Open-Vocabulary Object Detection (arxiv 2023) [[paper]](https://arxiv.org/abs/2312.15043) [[code]](https://github.com/om-ai-lab/GroundVLP)

- Context Disentangling and Prototype Inheriting for Robust Visual Grounding (TPAMI 2023) [[paper]](https://arxiv.org/abs/2312.11967) [[code]](https://github.com/WayneTomas/TransCP)

- Cycle-Consistency Learning for Captioning and Grounding (AAAI 2024) [[paper]](https://arxiv.org/abs/2312.15162)

- Zero-shot Referring Expression Comprehension via Structural Similarity Between Images and Captions (arxiv 2023) [[paper]](https://arxiv.org/abs/2311.17048) 

- Continual Referring Expression Comprehension via Dual Modular Memorization (arxiv 2023) [[paper]](https://arxiv.org/abs/2311.14909) [[code]](https://github.com/zackschen/DMM)

- ViLaM: A Vision-Language Model with Enhanced Visual Grounding and Generalization Capability (arxiv 2023) [[paper]](https://arxiv.org/abs/2311.12327)

- OV-VG: A Benchmark for Open-Vocabulary Visual Grounding (arxiv 2023) [[paper]](https://arxiv.org/abs/2310.14374) [[code]](https://github.com/cv516Buaa/OV-VG)

- VGDiffZero: Text-to-image Diffusion Models Can Be Zero-shot Visual Grounders (arxiv 2023) [[paper]](https://arxiv.org/abs/2309.01141)

- Language-Guided Diffusion Model for Visual Grounding (arxiv 2023) [[paper]](https://arxiv.org/abs/2308.09599) [[code (soon)]](https://github.com/iQua/vgbase/tree/DiffusionVG)

- Fine-Grained Visual Prompting (arxiv 2023) [[paper]](https://arxiv.org/abs/2306.04356)

- ONE-PEACE: Exploring One General Representation Model Toward Unlimited Modalities (arxiv 2023) [[paper]](https://arxiv.org/abs/2305.11172) [[code]](https://github.com/OFA-Sys/ONE-PEACE)

- CLIP-VG: Self-paced Curriculum Adapting of CLIP for Visual Grounding (TMM 2023) [[paper]](https://arxiv.org/abs/2305.08685) [[code]](https://github.com/linhuixiao/CLIP-VG)

- Unleashing Text-to-Image Diffusion Models for Visual Perception (ICCV 2023) [[paper]](https://arxiv.org/abs/2303.02153) [[website]](https://vpd.ivg-research.xyz/) [[code]](https://github.com/wl-zhao/VPD)

- Focusing On Targets For Improving Weakly Supervised Visual Grounding (ICASSP 2023) [[paper]](https://arxiv.org/abs/2302.11252)

- Unified-IO: A Unified Model for Vision, Language, and Multi-Modal Tasks (ICLR 2023) [[paper]](https://arxiv.org/abs/2206.08916) [[code (eval)]](https://github.com/allenai/unified-io-inference)

- PolyFormer: Referring Image Segmentation as Sequential Polygon Generation (CVPR 2023) [[paper]](https://arxiv.org/abs/2302.07387) [[website]](https://polyformer.github.io/) [[code]](https://github.com/amazon-science/polygon-transformer) [[demo]](https://huggingface.co/spaces/koajoel/PolyFormer)

- Advancing Visual Grounding With Scene Knowledge: Benchmark and Method (CVPR 2023) [[paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Song_Advancing_Visual_Grounding_With_Scene_Knowledge_Benchmark_and_Method_CVPR_2023_paper.pdf) [[code]](https://github.com/zhjohnchan/SK-VG)

- Language Adaptive Weight Generation for Multi-task Visual Grounding (CVPR 2023) [[paper]](https://arxiv.org/abs/2306.04652)

- From Coarse to Fine-grained Concept based Discrimination for Phrase Detection (CVPR 2023 Workshop) [[paper]](https://arxiv.org/abs/2112.03237)

- Referring Expression Comprehension Using Language Adaptive Inference (AAAI 2023) [[paper]](https://arxiv.org/abs/2306.04451)

- DQ-DETR: Dual Query Detection Transformer for Phrase Extraction and Grounding (AAAI 2023) [[paper]](https://arxiv.org/abs/2211.15516) [[code]](https://github.com/IDEA-Research/DQ-DETR)

- One for All: One-stage Referring Expression Comprehension with Dynamic Reasoning (arxiv 2022) [[paper]](https://arxiv.org/abs/2208.00361)

- Self-paced Multi-grained Cross-modal Interaction Modeling for Referring Expression Comprehension (arxiv 2022) [[paper]](https://arxiv.org/abs/2204.09957)

- SeqTR: A Simple yet Universal Network for Visual Grounding (ECCV 2022) [[paper]](https://arxiv.org/abs/2203.16265) [[code]](https://github.com/sean-zhuh/SeqTR)

- SiRi: A Simple Selective Retraining Mechanism for Transformer-based Visual Grounding (ECCV 2022) [[paper]](https://arxiv.org/abs/2207.13325)

- Towards Unifying Reference Expression Generation and Comprehension (EMNLP 2022) [[paper]](https://arxiv.org/abs/2210.13076)

- Correspondence Matters for Video Referring Expression Comprehension (ACM MM 2022) [[paper]](https://dl.acm.org/doi/abs/10.1145/3503161.3547756)

- Visual Grounding with Transformers (ICME 2022) [[paper]](https://arxiv.org/abs/2105.04281) [[code]](https://github.com/usr922/vgtr)

- Improving Visual Grounding with Visual-Linguistic Verification and Iterative Reasoning (CVPR 2022) [[paper]](https://arxiv.org/abs/2205.00272) [[code]](https://github.com/yangli18/VLTVG)

- Multi-Modal Dynamic Graph Transformer for Visual Grounding (CVPR 2022) [[paper]](https://github.com/iQua/M-DGT) [[code]](https://github.com/iQua/M-DGT)

- Pseudo-Q: Generating Pseudo Language Queries for Visual Grounding (CVPR 2022) [[paper]](https://arxiv.org/abs/2203.08481) [[code]](https://github.com/LeapLabTHU/Pseudo-Q)

- OFA: Unifying Architectures, Tasks, and Modalities Through a Simple Sequence-to-Sequence Learning Framework (ICML 2022) [[paper]](https://arxiv.org/abs/2202.03052) [[code]](https://github.com/OFA-Sys/OFA)

- Towards Language-guided Visual Recognition via Dynamic Convolutions (arxiv 2021) [[paper]](https://arxiv.org/abs/2110.08797)

- Referring Transformer: A One-step Approach to Multi-task Visual Grounding (NeurIPS 2021) [[paper]](https://arxiv.org/abs/2106.03089) [[code]](https://github.com/ubc-vision/RefTR)

- InstanceRefer: Cooperative Holistic Understanding for Visual Grounding on Point Clouds through Instance Multi-level Contextual Referring (ICCV 2021) [[paper]](https://openaccess.thecvf.com/content/ICCV2021/papers/Yuan_InstanceRefer_Cooperative_Holistic_Understanding_for_Visual_Grounding_on_Point_Clouds_ICCV_2021_paper.pdf) [[code]](https://github.com/CurryYuan/InstanceRefer)

- MDETR -- Modulated Detection for End-to-End Multi-Modal Understanding (ICCV 2021) [[paper]](https://arxiv.org/abs/2104.12763) [[website]](https://ashkamath.github.io/mdetr_page/) [[code]](https://github.com/ashkamath/mdetr)
<!-- ![Star](https://img.shields.io/github/stars/ashkamath/mdetr.svg?style=social&label=Star) -->

- Look Before You Leap: Learning Landmark Features for One-Stage Visual Grounding (CVPR 2021) [[paper]](https://arxiv.org/abs/2104.04386) [[code]](https://github.com/svip-lab/LBYLNet)

- Co-Grounding Networks with Semantic Attention for Referring Expression Comprehension in Videos (CVPR 2021) [[paper]](https://arxiv.org/abs/2103.12346) [[code]](https://github.com/SijieSong/CVPR21-Cogrounding_semantic_attention)

- Relation-aware Instance Refinement for Weakly Supervised Visual Grounding (CVPR 2021) [[paper]](https://arxiv.org/abs/2103.12989) [[code]](https://github.com/youngfly11/ReIR-WeaklyGrounding.pytorch)

- Large-Scale Adversarial Training for Vision-and-Language Representation Learning (NeurIPS 2020) [[paper]](https://arxiv.org/abs/2006.06195) [[code]](https://github.com/zhegan27/VILLA) [[poster]](https://zhegan27.github.io/Papers/villa_poster.pdf)

- Improving One-stage Visual Grounding by Recursive Sub-query Construction (ECCV 2020) [[paper]](https://arxiv.org/abs/2008.01059) [[code]](https://github.com/zyang-ur/ReSC)

- UNITER: UNiversal Image-TExt Representation Learning (ECCV 2020) [[paper]](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123750103.pdf) [[code]](https://github.com/ChenRocks/UNITER)

- Multi-task Collaborative Network for Joint Referring Expression Comprehension and Segmentation (CVPR 2020) [[paper]](https://arxiv.org/abs/2003.08813) [[code]](https://github.com/luogen1996/MCN)

- A Real-Time Cross-modality Correlation Filtering Method for Referring
Expression Comprehension (CVPR 2020) [[paper]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Liao_A_Real-Time_Cross-Modality_Correlation_Filtering_Method_for_Referring_Expression_Comprehension_CVPR_2020_paper.pdf)

- Dynamic Graph Attention for Referring Expression Comprehension (ICCV 2019) [[paper]](https://openaccess.thecvf.com/content_ICCV_2019/html/Yang_Dynamic_Graph_Attention_for_Referring_Expression_Comprehension_ICCV_2019_paper.html)

- A Fast and Accurate One-Stage Approach to Visual Grounding (ICCV 2019) [[paper]](https://arxiv.org/abs/1908.06354) [[code]](https://github.com/zyang-ur/onestage_grounding)

- Neighbourhood Watch: Referring Expression Comprehension via Language-Guided Graph Attention Networks (CVPR 2019) [[paper]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Neighbourhood_Watch_Referring_Expression_Comprehension_via_Language-Guided_Graph_Attention_Networks_CVPR_2019_paper.pdf)

- Interactive Visual Grounding of Referring Expressions for Human-Robot Interaction (RSS 2018) [[paper]](https://www.roboticsproceedings.org/rss14/p28.pdf) [[code]](https://github.com/MohitShridhar/ingress)

- Rethinking Diversified and Discriminative Proposal Generation for Visual Grounding (IJCAI 2018) [[paper]](https://arxiv.org/abs/1805.03508) [[code]](https://github.com/XiangChenchao/DDPN)

- MAttNet: Modular Attention Network for Referring Expression Comprehension (CVPR 2018) [[paper]](https://arxiv.org/abs/1801.08186) [[code]](https://github.com/lichengunc/MAttNet)

- Comprehension-Guided Referring Expressions (CVPR 2017) [[paper]](https://openaccess.thecvf.com/content_cvpr_2017/papers/Luo_Comprehension-Guided_Referring_Expressions_CVPR_2017_paper.pdf)

- Modeling Context Between Objects for Referring Expression Understanding (ECCV 2016) [[paper]](https://link.springer.com/chapter/10.1007/978-3-319-46493-0_48)


# Awesome Datasets

This part is still in progress.

## Datasets for DOD and Similar Tasks


| Name | Paper | Website | Code | Train/Eval | Notes |
|:-----|:-----:|:----:|:-----:|:-----:|:-----:|
| $D^3$ | [Described Object Detection: Liberating Object Detection with Flexible Expressions (NeurIPS 2023)](https://arxiv.org/abs/2307.12813) | - | [Github](https://github.com/shikras/d-cube) | eval only | - |


## Detection Datasets


| Name | Paper | Task | Website | Code | Train/Eval | Notes |
|:-----|:-----:|:----:|:-----:|:-----:|:-----:|:-----:|
| **Bamboo** | [Bamboo: Building Mega-Scale Vision Dataset Continually with Human-Machine Synergy](https://arxiv.org/abs/2203.07845) | OD | - | [Github](https://github.com/ZhangYuanhan-AI/Bamboo) | detector pretraining | build upon public datasets; 69M image classification annotations and 32M object bounding boxes |
| **BigDetection** | [BigDetection: A Large-scale Benchmark for Improved Object Detector Pre-training (CVPR 2022 workshop)](https://arxiv.org/abs/2203.13249) | OD | - | [Github](https://github.com/amazon-science/bigdetection) | detector pretraining | - |
| **Object365** | [Objects365: A Large-Scale, High-Quality Dataset for Object Detection (ICCV 2019)](https://openaccess.thecvf.com/content_ICCV_2019/html/Shao_Objects365_A_Large-Scale_High-Quality_Dataset_for_Object_Detection_ICCV_2019_paper.html) | OD | [Link](https://www.objects365.org) | [BAAI platform for download](https://data.baai.ac.cn/details/Objects365_2020) | detector pretraining; train & eval | - |
| **OpenImages** | - | OD | [Link](https://storage.googleapis.com/openimages/web/index.html) | [Tensorflow API](https://www.tensorflow.org/datasets/catalog/open_images_v4) | train & eval | - |
| **LVIS** | [LVIS: A Dataset for Large Vocabulary Instance Segmentation (CVPR 2019)](https://arxiv.org/abs/1908.03195) | OD&OVD | [Link](https://www.lvisdataset.org/) | [Github](https://github.com/lvis-dataset/lvis-api) | train & eval | long-tail; federated annotation; also used for OVD |
| **COCO** | [Microsoft COCO: Common Objects in Context (ECCV 2014)](https://arxiv.org/abs/1405.0312) | OD&OVD | [Link](https://cocodataset.org/#home) | [Github](https://github.com/cocodataset/cocoapi) | train & eval | also used for OVD |
| **VOC** | [The PASCAL Visual Object Classes (VOC) Challenge (IJCV 2010)](https://link.springer.com/article/10.1007/s11263-009-0275-4) | OD | [Link](http://host.robots.ox.ac.uk/pascal/VOC/index.html) | - | train & eval | - |


## Grounding Datasets
| Name | Paper | Task | Website | Code | Train/Eval | Notes |
|:-----|:-----:|:----:|:-----:|:-----:|:-----:|:-----:|
| **GRIT (Ground-and-Refer Instruction-Tuning)** | [Ferret: Refer and Ground Anything Anywhere at Any Granularity (arxiv 2023)](https://arxiv.org/abs/2310.07704) | ground-and-refer | - | [Github](https://github.com/apple/ml-ferret) | instruction tuning | 1.1M samples |
| **Ferret-Bench** | [Ferret: Refer and Ground Anything Anywhere at Any Granularity (arxiv 2023)](https://arxiv.org/abs/2310.07704) | ground-and-refer | - | [Github](https://github.com/apple/ml-ferret) | eval only | - |
| **GRIT (Grounded Image-Text)** | [Kosmos-2: Grounding Multimodal Large Language Models to the World (arxiv 2023)](https://arxiv.org/abs/2306.14824) | visual grounding (REC & Phrase Grounding) | - | [Github](https://github.com/microsoft/unilm/tree/master/kosmos-2#grit-large-scale-training-corpus-of-grounded-image-text-pairs) [Huggingface](https://huggingface.co/datasets/zzliang/GRIT) | train only | created based on image-text pairs from a subset of COYO-700M and LAION-2B; 20.5M |
| **SK-VG** | [Advancing Visual Grounding With Scene Knowledge: Benchmark and Method (CVPR 2023)](https://openaccess.thecvf.com/content/CVPR2023/papers/Song_Advancing_Visual_Grounding_With_Scene_Knowledge_Benchmark_and_Method_CVPR_2023_paper.pdf) | REC | - | [Github](https://github.com/zhjohnchan/SK-VG) | train & eval | scene knowledge in natural language is required |
| **GRiT (General Robust Image Task)** | [GRIT: General Robust Image Task Benchmark (arxiv 2022)](https://arxiv.org/abs/2204.13653) | REC | [Link](https://allenai.org/project/grit/home) | [Github](https://github.com/allenai/grit_official) | eval only | - |
| **Cops-Ref** | [Cops-Ref: A new Dataset and Task on Compositional Referring Expression Comprehension (CVPR 2020)](https://arxiv.org/abs/2003.00403) | Compositional REC | - | [Github](https://github.com/zfchenUnique/Cops-Ref) | eval only | A variant of REC |
| **Visual Genome** | [Visual Genome: Connecting Language and Vision Using Crowdsourced Dense Image Annotations (IJCV 2017)](https://link.springer.com/article/10.1007/s11263-016-0981-7) | OD & Phrase Grounding | [Link](https://homes.cs.washington.edu/~ranjay/visualgenome/index.html) | [Github](https://github.com/ranjaykrishna/visual_genome_python_driver) | multiple multi-modal tasks (including REC) |
| **RefCOCOg** | [Generation and Comprehension of Unambiguous Object Descriptions (CVPR 2016)](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Mao_Generation_and_Comprehension_CVPR_2016_paper.pdf) | REC |  - | [Github](https://github.com/mjhucla/Google_Refexp_toolbox) | train & eval | images from COCO |
| **RefClef** | [ReferItGame: Referring to Objects in Photographs of Natural Scenes (EMNLP 2014)](https://arxiv.org/abs/2204.13653) | REC | - | [Github](https://github.com/lichengunc/refer) | train & eval | - |
| **RefCOCO+** | [ReferItGame: Referring to Objects in Photographs of Natural Scenes (EMNLP 2014)](https://arxiv.org/abs/2204.13653) | REC | - | [Github](https://github.com/lichengunc/refer) | train & eval | images from COCO |
| **RefCOCO** | [ReferItGame: Referring to Objects in Photographs of Natural Scenes (EMNLP 2014)](https://arxiv.org/abs/2204.13653) | REC | - | [Github](https://github.com/lichengunc/refer) | train & eval | images from COCO |


# Related Surveys and Resources

Some survey papers regarding relevant tasks (open-vocabulary learning, etc.)

- Towards Open Vocabulary Learning: A Survey (arxiv 2023) [[paper]](https://arxiv.org/abs/2306.15880) [[repo]](https://github.com/jianzongwu/Awesome-Open-Vocabulary)
- A Survey on Open-Vocabulary Detection and Segmentation: Past, Present, and Future (arxiv 2023) [[paper]](https://arxiv.org/abs/2307.09220)
- Referring Expression Comprehension: A Survey of Methods and Datasets (TMM 2020) [[paper]](https://arxiv.org/abs/2007.09554)

Some similar github repos like awesome lists:

- [daqingliu/awesome-rec](https://github.com/daqingliu/awesome-rec): A curated list of REC papers. Not maintained in recent years.
- [qy-feng/awesome-visual-grounding](https://github.com/qy-feng/awesome-visual-grounding): A curated list of visual grounding papers. Not maintained in recent years.
- [MarkMoHR/Awesome-Referring-Image-Segmentation](https://github.com/MarkMoHR/Awesome-Referring-Image-Segmentation): A list of Referring Expression Segmentation (RES) papers and resources.
- [TheShadow29/awesome-grounding](https://github.com/TheShadow29/awesome-grounding): A list of visual grounding (REC) paper roadmaps and datasets.
- [witnessai/Awesome-Open-Vocabulary-Object-Detection](https://github.com/witnessai/Awesome-Open-Vocabulary-Object-Detection/blob/main/README.md?plain=1): A list of Open-Vocabulary Object Detection papers.



# Acknowledgement

The structure and format of this repo is inspired by [BradyFU/Awesome-Multimodal-Large-Language-Models](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models).
