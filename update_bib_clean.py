import re
from pathlib import Path

bib_path = Path("docs/paper/latex/custom.bib")
text = bib_path.read_text(encoding="utf-8")

# 1. Chen 2023
old_chen2023 = r"""@article{chen2023accelerating,
  title     = {Accelerating Large Language Model Decoding with Speculative Sampling},
  author    = {Chen, Charlie and Borgeaud, Sebastian and O'Donoghue, Alistair and Zhou, Alexey and Cassirer, Roman and Roux, Nicolas and Irving, Geoffrey and Rae, Jack W.},
  journal   = {arXiv preprint arXiv:2302.01318},
  year      = {2023}
}"""
new_chen2023 = r"""@article{chen2023accelerating,
  title     = {Accelerating Large Language Model Decoding with Speculative Sampling},
  author    = {Chen, Charlie and Borgeaud, Sebastian and Irving, Geoffrey and Lespiau, Jean-Baptiste and Sifre, Laurent and Jumper, John},
  journal   = {arXiv preprint arXiv:2302.01318},
  year      = {2023}
}"""
text = text.replace(old_chen2023, new_chen2023)

# 2. MagicDec
old_magicdec = r"""@article{chen2024magicdec,
  title     = {{MagicDec}: Breaking the Latency-Throughput Tradeoff for Long Context Generation with Speculative Decoding},
  author    = {Chen, Jian and Tiwari, Vashisth and Zhang, Ranran Haoran and others},
  journal   = {arXiv preprint arXiv:2408.11049},
  year      = {2024}
}"""
new_magicdec = r"""@article{chen2024magicdec,
  title     = {{MagicDec}: Breaking the Latency-Throughput Tradeoff for Long Context Generation with Speculative Decoding},
  author    = {Sadhukhan, Ranajoy and Chen, Jian and Tiwari, Vashisth and Chen, Zhuoming and Shi, Jinyuan and Yen, Ian En-Hsu and Chen, Beidi},
  journal   = {arXiv preprint arXiv:2408.11049},
  year      = {2024}
}"""
text = text.replace(old_magicdec, new_magicdec)

# 3. TriForce
old_triforce = r"""@inproceedings{sun2024triforce,
  title     = {{TriForce}: Lossless Acceleration of Long Sequence Generation with Hierarchical Speculative Decoding},
  author    = {Sun, Hanshi and Chen, Zhuoming and Cheng, Xinhao and Zhang, Huishuai and Zhang, Zheng and Chen, Beidi},
  booktitle = {Proceedings of the 41st International Conference on Machine Learning (ICML)},
  year      = {2024}
}"""
new_triforce = r"""@inproceedings{sun2024triforce,
  title     = {{TriForce}: Lossless Acceleration of Long Sequence Generation with Hierarchical Speculative Decoding},
  author    = {Sun, Hanshi and Chen, Zhuoming and Yang, Xinyu and Tian, Yuandong and Chen, Beidi},
  booktitle = {Proceedings of the 41st International Conference on Machine Learning (ICML)},
  year      = {2024}
}"""
text = text.replace(old_triforce, new_triforce)

# 4. SnapKV
old_snapkv = r"""@article{li2024snapkv,
  title     = {{SnapKV}: {LLM} Knows What You Are Looking for Before Generation},
  author    = {Li, Yuhong and Huang, Yingbing and Yang, Bowen and Venkitesh, Bharat and Liu, Aojun and others},
  journal   = {arXiv preprint arXiv:2404.14469},
  year      = {2024}
}"""
new_snapkv = r"""@article{li2024snapkv,
  title     = {{SnapKV}: {LLM} Knows What You Are Looking for Before Generation},
  author    = {Li, Yuhong and Huang, Yingbing and Yang, Bowen and Venkitesh, Bharat and Locatelli, Acyr and Ye, Hanchen and Cai, Tianle and Lewis, Patrick and Chen, Deming},
  journal   = {arXiv preprint arXiv:2404.14469},
  year      = {2024}
}"""
text = text.replace(old_snapkv, new_snapkv)

# 5. Prompt Lookup
old_promptlookup = r"""@misc{saxena2023promptlookup,
  title     = {Prompt Lookup Decoding},
  author    = {Saxena, Apuroop},
  howpublished = {\url{https://github.com/apoorvumang/prompt-lookup-decoding}},
  year      = {2023}
}"""
new_promptlookup = r"""@misc{saxena2023promptlookup,
  title     = {Prompt Lookup Decoding},
  author    = {Saxena, Apoorv},
  howpublished = {\url{https://github.com/apoorvumang/prompt-lookup-decoding}},
  year      = {2023}
}"""
text = text.replace(old_promptlookup, new_promptlookup)

# 6. RAPID
old_rapid = r"""@inproceedings{chen2025rapid,
  title     = {{RAPID}: Retrieval-Augmented Speculative Decoding for Efficient Long-Context Generation},
  author    = {Chen, Charlie and others},
  booktitle = {Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (ACL)},
  year      = {2025}
}"""
new_rapid = r"""@inproceedings{chen2025rapid,
  title     = {{RAPID}: Long-Context Inference with Retrieval-Augmented Speculative Decoding},
  author    = {Chen, Guanzheng and Shieh, Michael Qizhe and others},
  booktitle = {Proceedings of the 42nd International Conference on Machine Learning (ICML)},
  year      = {2025}
}"""
text = text.replace(old_rapid, new_rapid)

# 7. LongSpec
old_longspec = r"""@inproceedings{yang2026longspec,
  title     = {{LongSpec}: Long-Context Lossless Speculative Decoding with Efficient Drafting and Verification},
  author    = {Yang, Penghui and Du, Cunxiao and Zhang, Fengzhuo and Wang, Haonan and Pang, Tianyu and Du, Chao and An, Bo},
  booktitle = {Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (ACL)},
  year      = {2026}
}"""
new_longspec = r"""@article{yang2025longspec,
  title     = {{LongSpec}: Long-Context Lossless Speculative Decoding with Efficient Drafting and Verification},
  author    = {Yang, Penghui and Du, Cunxiao and Zhang, Fengzhuo and Wang, Haonan and Pang, Tianyu and Du, Chao and An, Bo},
  journal   = {arXiv preprint arXiv:2502.17421},
  year      = {2025}
}"""
text = text.replace(old_longspec, new_longspec)

# 8. LinguaSpec
old_linguaspec = r"""@inproceedings{liu2026linguaspec,
  title     = {{LinguaSpec}: Training-Free Adaptive Speculative Decoding via Linguistic Priors},
  author    = {Liu, Yuhan and others},
  booktitle = {Findings of the Association for Computational Linguistics: ACL 2026},
  year      = {2026}
}"""
new_linguaspec = r"""@inproceedings{wang2026linguaspec,
  title     = {{LinguaSpec}: Training-Free Adaptive Speculative Decoding via Linguistic Priors},
  author    = {Wang, Jingyi and Huang, Jiaqi and Xu, Zunnan and Zhou, Jun and Yuan, Kehong and Qian, Xiang},
  booktitle = {Findings of the Association for Computational Linguistics: ACL 2026},
  year      = {2026}
}"""
text = text.replace(old_linguaspec, new_linguaspec)

# 9. EDSD
old_edsd = r"""@inproceedings{wang2026edsd,
  title     = {{EDSD}: Entropy-Driven Design for Faster Speculative Decoding},
  author    = {Wang, Haoran and others},
  booktitle = {Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (ACL)},
  year      = {2026}
}"""
new_edsd = r"""@inproceedings{cheng2026edsd,
  title     = {{EDSD}: Entropy-Driven Design for Faster Speculative Decoding},
  author    = {Cheng, Longkai and Wang, Ximing and Zhu, Jiangcai and Shao, Kailai and Chen, Chao and Hu, Haixiang},
  booktitle = {Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (ACL)},
  year      = {2026}
}"""
text = text.replace(old_edsd, new_edsd)

# 10. HeteroSpec
old_heterospec = r"""@inproceedings{xu2026heterospec,
  title     = {{HeteroSpec}: Leveraging Contextual Heterogeneity for Efficient Speculative Decoding},
  author    = {Xu, Ming and others},
  booktitle = {Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (ACL)},
  year      = {2026}
}"""
new_heterospec = r"""@inproceedings{liu2026heterospec,
  title     = {{HeteroSpec}: Leveraging Contextual Heterogeneity for Efficient Speculative Decoding},
  author    = {Liu, Siran and Ye, Yang and Zhu, Qianchao and Cao, Zane and He, Yongchao},
  booktitle = {Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (ACL)},
  year      = {2026}
}"""
text = text.replace(old_heterospec, new_heterospec)

bib_path.write_text(text, encoding="utf-8")
print("Updated custom.bib successfully!")
