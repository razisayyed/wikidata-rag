# RAG vs BASELINE Benchmark Report

Generated: 2026-02-14T13:20:17
Analysis Version: `v1_legacy_simple`

## Run Configuration

- Threshold: `0.5`
- Temperature: `0.0`
- Total Cases: `30`

## Head-to-Head by Evaluator

| Evaluator | RAG Wins | BASELINE Wins | Ties | Skipped | Errors |
|---|---:|---:|---:|---:|---:|
| vectara | 12 | 1 | 17 | 0 | 0 |
| aimon | 11 | 5 | 14 | 0 | 0 |
| llm_judge | 10 | 1 | 19 | 0 | 0 |
| ragtruth | 13 | 4 | 13 | 0 | 0 |

## vectara Results

| # | Case ID | Question | RAG | BASELINE | RAG Score | BASELINE Score | Winner | Status |
|---:|---|---|---|---|---:|---:|---|---|
| 1 | multi_hop_001 | What capital corresponds to the country where Frida Kahlo was born? | factual | hallucinated | 1.000 | 0.000 | RAG | completed |
| 2 | multi_hop_002 | For The Great Gatsby, what is the capital of its author's citizenship country? | factual | factual | 1.000 | 1.000 | Tie | completed |
| 3 | multi_hop_003 | What is the capital of the nation where Microsoft's founder was born? | factual | hallucinated | 1.000 | 0.000 | RAG | completed |
| 4 | multi_hop_004 | Identify the continent of Karlovac Gymnasium's country for Nikola Tesla. | factual | factual | 1.000 | 1.000 | Tie | completed |
| 5 | multi_hop_005 | Which capital is tied to the country of birth of Mexico's head of state? | hallucinated | hallucinated | 1.000 | 1.000 | Tie | completed |
| 6 | multi_hop_006 | Identify the capital of the nation associated with Congressional Gold Medal that George Washington received. | factual | factual | 1.000 | 1.000 | Tie | completed |
| 7 | multi_hop_007 | Which capital belongs to the country of Ada Lovelace's birthplace? | factual | hallucinated | 1.000 | 1.000 | RAG | completed |
| 8 | multi_hop_008 | What capital is associated with the author's country in Don Quixote? | factual | factual | 1.000 | 1.000 | Tie | completed |
| 9 | multi_hop_009 | What is the capital of the country where the founder of Siemens was born? | factual | hallucinated | 1.000 | 0.000 | RAG | completed |
| 10 | multi_hop_010 | Identify the continent of Science Faculty of Paris's country for Marie Curie. | factual | hallucinated | 1.000 | 1.000 | RAG | completed |
| 11 | multi_hop_011 | What is the capital of the country where the head of state of South Africa was born? | hallucinated | hallucinated | 0.000 | 0.000 | Tie | completed |
| 12 | multi_hop_012 | Alan Turing received Officer of the Order of the British Empire; what is the capital of the award's country? | factual | factual | 1.000 | 1.000 | Tie | completed |
| 13 | multi_hop_013 | The birthplace of George Washington is in which country's capital city? | hallucinated | hallucinated | 1.000 | 1.000 | Tie | completed |
| 14 | multi_hop_014 | For Don Quixote, what is the capital of its author's citizenship country? | hallucinated | hallucinated | 1.000 | 0.000 | RAG | completed |
| 15 | multi_hop_015 | What is the capital of the country where the founder of Meta was born? | hallucinated | hallucinated | 0.000 | 0.000 | Tie | completed |
| 16 | multi_hop_016 | Identify the continent of Massachusetts Institute of Technology's country for Richard Feynman. | factual | hallucinated | 1.000 | 1.000 | RAG | completed |
| 17 | multi_hop_017 | Name the capital city of the birth country of Brazil's head of state. | hallucinated | factual | 1.000 | 1.000 | BASELINE | completed |
| 18 | multi_hop_018 | Isaac Newton received Knight Bachelor; what is the capital of the award's country? | factual | factual | 1.000 | 1.000 | Tie | completed |
| 19 | multi_hop_019 | What is the capital of the country where Charles Darwin was born? | factual | factual | 1.000 | 1.000 | Tie | completed |
| 20 | multi_hop_020 | Identify the capital of the nation linked to the author of Moby-Dick. | factual | hallucinated | 1.000 | 1.000 | RAG | completed |
| 21 | multi_hop_021 | Which capital corresponds to the founder-birth country of Cisco? | factual | hallucinated | 1.000 | 1.000 | RAG | completed |
| 22 | multi_hop_022 | The country of Albert Einstein's alma mater, ETH Zurich, lies on what continent? | factual | factual | 1.000 | 1.000 | Tie | completed |
| 23 | multi_hop_023 | What is the capital of the country where the head of state of Australia was born? | hallucinated | hallucinated | 0.000 | 0.000 | Tie | completed |
| 24 | multi_hop_024 | Charles Darwin received Copley Medal; what is the capital of the award's country? | factual | factual | 1.000 | 1.000 | Tie | completed |
| 25 | multi_hop_025 | What capital corresponds to the country where Nelson Mandela was born? | hallucinated | hallucinated | 0.000 | 0.000 | Tie | completed |
| 26 | multi_hop_026 | Name the capital city of the author's country for The Brothers Karamazov. | hallucinated | hallucinated | 0.000 | 0.000 | Tie | completed |
| 27 | multi_hop_027 | Which capital corresponds to the founder-birth country of Samsung Electronics? | factual | hallucinated | 1.000 | 1.000 | RAG | completed |
| 28 | multi_hop_028 | Identify the continent of Charles University's country for Nikola Tesla. | hallucinated | hallucinated | 1.000 | 1.000 | Tie | completed |
| 29 | multi_hop_029 | For United Kingdom, which capital belongs to its head of state's birth country? | factual | hallucinated | 1.000 | 0.000 | RAG | completed |
| 30 | multi_hop_030 | Name the capital city of the country linked to Grand Cross of the Order of the Sun of Peru‎, received by Angela Merkel. | factual | hallucinated | 1.000 | 1.000 | RAG | completed |

## aimon Results

| # | Case ID | Question | RAG | BASELINE | RAG Score | BASELINE Score | Winner | Status |
|---:|---|---|---|---|---:|---:|---|---|
| 1 | multi_hop_001 | What capital corresponds to the country where Frida Kahlo was born? | hallucinated | factual | 0.590 | 0.132 | BASELINE | completed |
| 2 | multi_hop_002 | For The Great Gatsby, what is the capital of its author's citizenship country? | factual | factual | 0.045 | 0.064 | Tie | completed |
| 3 | multi_hop_003 | What is the capital of the nation where Microsoft's founder was born? | factual | hallucinated | 0.056 | 0.984 | RAG | completed |
| 4 | multi_hop_004 | Identify the continent of Karlovac Gymnasium's country for Nikola Tesla. | factual | factual | 0.021 | 0.051 | Tie | completed |
| 5 | multi_hop_005 | Which capital is tied to the country of birth of Mexico's head of state? | hallucinated | hallucinated | 0.992 | 0.992 | Tie | completed |
| 6 | multi_hop_006 | Identify the capital of the nation associated with Congressional Gold Medal that George Washington received. | factual | factual | 0.086 | 0.086 | Tie | completed |
| 7 | multi_hop_007 | Which capital belongs to the country of Ada Lovelace's birthplace? | factual | hallucinated | 0.291 | 0.973 | RAG | completed |
| 8 | multi_hop_008 | What capital is associated with the author's country in Don Quixote? | factual | factual | 0.005 | 0.004 | Tie | completed |
| 9 | multi_hop_009 | What is the capital of the country where the founder of Siemens was born? | factual | hallucinated | 0.116 | 0.852 | RAG | completed |
| 10 | multi_hop_010 | Identify the continent of Science Faculty of Paris's country for Marie Curie. | factual | hallucinated | 0.305 | 0.980 | RAG | completed |
| 11 | multi_hop_011 | What is the capital of the country where the head of state of South Africa was born? | hallucinated | hallucinated | 0.969 | 0.969 | Tie | completed |
| 12 | multi_hop_012 | Alan Turing received Officer of the Order of the British Empire; what is the capital of the award's country? | factual | factual | 0.126 | 0.130 | Tie | completed |
| 13 | multi_hop_013 | The birthplace of George Washington is in which country's capital city? | hallucinated | hallucinated | 0.984 | 0.984 | Tie | completed |
| 14 | multi_hop_014 | For Don Quixote, what is the capital of its author's citizenship country? | hallucinated | hallucinated | 0.984 | 0.867 | BASELINE | completed |
| 15 | multi_hop_015 | What is the capital of the country where the founder of Meta was born? | hallucinated | hallucinated | 0.938 | 0.922 | Tie | completed |
| 16 | multi_hop_016 | Identify the continent of Massachusetts Institute of Technology's country for Richard Feynman. | hallucinated | hallucinated | 0.875 | 0.992 | RAG | completed |
| 17 | multi_hop_017 | Name the capital city of the birth country of Brazil's head of state. | hallucinated | factual | 0.992 | 0.471 | BASELINE | completed |
| 18 | multi_hop_018 | Isaac Newton received Knight Bachelor; what is the capital of the award's country? | factual | factual | 0.106 | 0.121 | Tie | completed |
| 19 | multi_hop_019 | What is the capital of the country where Charles Darwin was born? | factual | factual | 0.169 | 0.116 | BASELINE | completed |
| 20 | multi_hop_020 | Identify the capital of the nation linked to the author of Moby-Dick. | factual | hallucinated | 0.104 | 0.980 | RAG | completed |
| 21 | multi_hop_021 | Which capital corresponds to the founder-birth country of Cisco? | factual | hallucinated | 0.441 | 0.980 | RAG | completed |
| 22 | multi_hop_022 | The country of Albert Einstein's alma mater, ETH Zurich, lies on what continent? | factual | factual | 0.013 | 0.166 | RAG | completed |
| 23 | multi_hop_023 | What is the capital of the country where the head of state of Australia was born? | hallucinated | hallucinated | 0.969 | 0.969 | Tie | completed |
| 24 | multi_hop_024 | Charles Darwin received Copley Medal; what is the capital of the award's country? | factual | factual | 0.098 | 0.055 | Tie | completed |
| 25 | multi_hop_025 | What capital corresponds to the country where Nelson Mandela was born? | hallucinated | hallucinated | 0.949 | 0.824 | BASELINE | completed |
| 26 | multi_hop_026 | Name the capital city of the author's country for The Brothers Karamazov. | hallucinated | hallucinated | 0.914 | 0.902 | Tie | completed |
| 27 | multi_hop_027 | Which capital corresponds to the founder-birth country of Samsung Electronics? | factual | hallucinated | 0.225 | 0.992 | RAG | completed |
| 28 | multi_hop_028 | Identify the continent of Charles University's country for Nikola Tesla. | hallucinated | hallucinated | 0.977 | 0.977 | Tie | completed |
| 29 | multi_hop_029 | For United Kingdom, which capital belongs to its head of state's birth country? | factual | hallucinated | 0.348 | 0.941 | RAG | completed |
| 30 | multi_hop_030 | Name the capital city of the country linked to Grand Cross of the Order of the Sun of Peru‎, received by Angela Merkel. | hallucinated | hallucinated | 0.516 | 0.980 | RAG | completed |

## llm_judge Results

| # | Case ID | Question | RAG | BASELINE | RAG Score | BASELINE Score | Winner | Status |
|---:|---|---|---|---|---:|---:|---|---|
| 1 | multi_hop_001 | What capital corresponds to the country where Frida Kahlo was born? | factual | factual | N/A | N/A | Tie | completed |
| 2 | multi_hop_002 | For The Great Gatsby, what is the capital of its author's citizenship country? | factual | factual | N/A | N/A | Tie | completed |
| 3 | multi_hop_003 | What is the capital of the nation where Microsoft's founder was born? | factual | hallucinated | N/A | N/A | RAG | completed |
| 4 | multi_hop_004 | Identify the continent of Karlovac Gymnasium's country for Nikola Tesla. | factual | factual | N/A | N/A | Tie | completed |
| 5 | multi_hop_005 | Which capital is tied to the country of birth of Mexico's head of state? | hallucinated | hallucinated | N/A | N/A | Tie | completed |
| 6 | multi_hop_006 | Identify the capital of the nation associated with Congressional Gold Medal that George Washington received. | factual | factual | N/A | N/A | Tie | completed |
| 7 | multi_hop_007 | Which capital belongs to the country of Ada Lovelace's birthplace? | factual | hallucinated | N/A | N/A | RAG | completed |
| 8 | multi_hop_008 | What capital is associated with the author's country in Don Quixote? | factual | factual | N/A | N/A | Tie | completed |
| 9 | multi_hop_009 | What is the capital of the country where the founder of Siemens was born? | factual | hallucinated | N/A | N/A | RAG | completed |
| 10 | multi_hop_010 | Identify the continent of Science Faculty of Paris's country for Marie Curie. | factual | hallucinated | N/A | N/A | RAG | completed |
| 11 | multi_hop_011 | What is the capital of the country where the head of state of South Africa was born? | hallucinated | hallucinated | N/A | N/A | Tie | completed |
| 12 | multi_hop_012 | Alan Turing received Officer of the Order of the British Empire; what is the capital of the award's country? | factual | factual | N/A | N/A | Tie | completed |
| 13 | multi_hop_013 | The birthplace of George Washington is in which country's capital city? | hallucinated | hallucinated | N/A | N/A | Tie | completed |
| 14 | multi_hop_014 | For Don Quixote, what is the capital of its author's citizenship country? | hallucinated | hallucinated | N/A | N/A | Tie | completed |
| 15 | multi_hop_015 | What is the capital of the country where the founder of Meta was born? | hallucinated | hallucinated | N/A | N/A | Tie | completed |
| 16 | multi_hop_016 | Identify the continent of Massachusetts Institute of Technology's country for Richard Feynman. | hallucinated | hallucinated | N/A | N/A | Tie | completed |
| 17 | multi_hop_017 | Name the capital city of the birth country of Brazil's head of state. | hallucinated | factual | N/A | N/A | BASELINE | completed |
| 18 | multi_hop_018 | Isaac Newton received Knight Bachelor; what is the capital of the award's country? | factual | factual | N/A | N/A | Tie | completed |
| 19 | multi_hop_019 | What is the capital of the country where Charles Darwin was born? | factual | factual | N/A | N/A | RAG | completed |
| 20 | multi_hop_020 | Identify the capital of the nation linked to the author of Moby-Dick. | factual | hallucinated | N/A | N/A | RAG | completed |
| 21 | multi_hop_021 | Which capital corresponds to the founder-birth country of Cisco? | factual | hallucinated | N/A | N/A | RAG | completed |
| 22 | multi_hop_022 | The country of Albert Einstein's alma mater, ETH Zurich, lies on what continent? | factual | factual | N/A | N/A | Tie | completed |
| 23 | multi_hop_023 | What is the capital of the country where the head of state of Australia was born? | hallucinated | hallucinated | N/A | N/A | Tie | completed |
| 24 | multi_hop_024 | Charles Darwin received Copley Medal; what is the capital of the award's country? | factual | factual | N/A | N/A | Tie | completed |
| 25 | multi_hop_025 | What capital corresponds to the country where Nelson Mandela was born? | hallucinated | hallucinated | N/A | N/A | Tie | completed |
| 26 | multi_hop_026 | Name the capital city of the author's country for The Brothers Karamazov. | hallucinated | hallucinated | N/A | N/A | Tie | completed |
| 27 | multi_hop_027 | Which capital corresponds to the founder-birth country of Samsung Electronics? | factual | hallucinated | N/A | N/A | RAG | completed |
| 28 | multi_hop_028 | Identify the continent of Charles University's country for Nikola Tesla. | hallucinated | hallucinated | N/A | N/A | Tie | completed |
| 29 | multi_hop_029 | For United Kingdom, which capital belongs to its head of state's birth country? | factual | hallucinated | N/A | N/A | RAG | completed |
| 30 | multi_hop_030 | Name the capital city of the country linked to Grand Cross of the Order of the Sun of Peru‎, received by Angela Merkel. | factual | hallucinated | N/A | N/A | RAG | completed |

## ragtruth Results

| # | Case ID | Question | RAG | BASELINE | RAG Score | BASELINE Score | Winner | Status |
|---:|---|---|---|---|---:|---:|---|---|
| 1 | multi_hop_001 | What capital corresponds to the country where Frida Kahlo was born? | factual | factual | 0.000 | 0.000 | Tie | completed |
| 2 | multi_hop_002 | For The Great Gatsby, what is the capital of its author's citizenship country? | factual | factual | 0.000 | 0.000 | Tie | completed |
| 3 | multi_hop_003 | What is the capital of the nation where Microsoft's founder was born? | factual | hallucinated | 0.000 | 0.139 | RAG | completed |
| 4 | multi_hop_004 | Identify the continent of Karlovac Gymnasium's country for Nikola Tesla. | factual | factual | 0.000 | 0.000 | Tie | completed |
| 5 | multi_hop_005 | Which capital is tied to the country of birth of Mexico's head of state? | hallucinated | hallucinated | 0.000 | 0.000 | Tie | completed |
| 6 | multi_hop_006 | Identify the capital of the nation associated with Congressional Gold Medal that George Washington received. | factual | factual | 0.000 | 0.000 | Tie | completed |
| 7 | multi_hop_007 | Which capital belongs to the country of Ada Lovelace's birthplace? | hallucinated | hallucinated | 0.478 | 0.000 | BASELINE | completed |
| 8 | multi_hop_008 | What capital is associated with the author's country in Don Quixote? | factual | factual | 0.000 | 0.000 | Tie | completed |
| 9 | multi_hop_009 | What is the capital of the country where the founder of Siemens was born? | factual | hallucinated | 0.000 | 0.057 | RAG | completed |
| 10 | multi_hop_010 | Identify the continent of Science Faculty of Paris's country for Marie Curie. | factual | hallucinated | 0.000 | 0.000 | RAG | completed |
| 11 | multi_hop_011 | What is the capital of the country where the head of state of South Africa was born? | hallucinated | hallucinated | 0.324 | 0.485 | RAG | completed |
| 12 | multi_hop_012 | Alan Turing received Officer of the Order of the British Empire; what is the capital of the award's country? | factual | factual | 0.000 | 0.000 | Tie | completed |
| 13 | multi_hop_013 | The birthplace of George Washington is in which country's capital city? | hallucinated | hallucinated | 0.000 | 0.000 | Tie | completed |
| 14 | multi_hop_014 | For Don Quixote, what is the capital of its author's citizenship country? | hallucinated | hallucinated | 0.000 | 0.149 | RAG | completed |
| 15 | multi_hop_015 | What is the capital of the country where the founder of Meta was born? | hallucinated | hallucinated | 0.195 | 0.152 | BASELINE | completed |
| 16 | multi_hop_016 | Identify the continent of Massachusetts Institute of Technology's country for Richard Feynman. | hallucinated | hallucinated | 0.564 | 0.555 | BASELINE | completed |
| 17 | multi_hop_017 | Name the capital city of the birth country of Brazil's head of state. | hallucinated | hallucinated | 0.000 | 0.718 | RAG | completed |
| 18 | multi_hop_018 | Isaac Newton received Knight Bachelor; what is the capital of the award's country? | factual | factual | 0.000 | 0.000 | Tie | completed |
| 19 | multi_hop_019 | What is the capital of the country where Charles Darwin was born? | factual | factual | 0.000 | 0.000 | Tie | completed |
| 20 | multi_hop_020 | Identify the capital of the nation linked to the author of Moby-Dick. | factual | hallucinated | 0.000 | 0.000 | RAG | completed |
| 21 | multi_hop_021 | Which capital corresponds to the founder-birth country of Cisco? | factual | hallucinated | 0.000 | 0.000 | RAG | completed |
| 22 | multi_hop_022 | The country of Albert Einstein's alma mater, ETH Zurich, lies on what continent? | factual | factual | 0.000 | 0.000 | Tie | completed |
| 23 | multi_hop_023 | What is the capital of the country where the head of state of Australia was born? | hallucinated | hallucinated | 1.000 | 0.332 | BASELINE | completed |
| 24 | multi_hop_024 | Charles Darwin received Copley Medal; what is the capital of the award's country? | factual | factual | 0.000 | 0.000 | Tie | completed |
| 25 | multi_hop_025 | What capital corresponds to the country where Nelson Mandela was born? | hallucinated | hallucinated | 0.107 | 0.442 | RAG | completed |
| 26 | multi_hop_026 | Name the capital city of the author's country for The Brothers Karamazov. | hallucinated | hallucinated | 0.052 | 0.077 | RAG | completed |
| 27 | multi_hop_027 | Which capital corresponds to the founder-birth country of Samsung Electronics? | factual | hallucinated | 0.000 | 0.000 | RAG | completed |
| 28 | multi_hop_028 | Identify the continent of Charles University's country for Nikola Tesla. | hallucinated | hallucinated | 0.000 | 0.000 | Tie | completed |
| 29 | multi_hop_029 | For United Kingdom, which capital belongs to its head of state's birth country? | factual | hallucinated | 0.000 | 0.000 | RAG | completed |
| 30 | multi_hop_030 | Name the capital city of the country linked to Grand Cross of the Order of the Sun of Peru‎, received by Angela Merkel. | factual | hallucinated | 0.000 | 0.000 | RAG | completed |

## Skipped/Error Diagnostics

- None