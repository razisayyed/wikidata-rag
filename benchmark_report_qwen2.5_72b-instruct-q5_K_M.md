# RAG vs BASELINE Benchmark Report

Generated: 2026-02-15T12:48:41
Analysis Version: `v1_legacy_simple`

## Run Configuration

- Threshold: `0.5`
- Temperature: `0.0`
- Total Cases: `80`

## Head-to-Head by Evaluator

| Evaluator | RAG Wins | BASELINE Wins | Ties | Skipped | Errors |
|---|---:|---:|---:|---:|---:|
| vectara | 34 | 10 | 36 | 0 | 0 |
| aimon | 53 | 11 | 16 | 0 | 0 |
| llm_judge | 41 | 5 | 34 | 0 | 0 |
| ragtruth | 43 | 18 | 19 | 0 | 0 |

## Aggregated Metrics

### Factual vs Hallucinated

| Evaluator | RAG Factual | RAG Hallucinated | BASELINE Factual | BASELINE Hallucinated |
|---|---:|---:|---:|---:|
| vectara | 63 | 17 | 35 | 45 |
| aimon | 50 | 30 | 29 | 51 |
| llm_judge | 62 | 18 | 32 | 48 |
| ragtruth | 56 | 24 | 23 | 57 |

### Completion Status

| Evaluator | Completed | Skipped | Errors |
|---|---:|---:|---:|
| vectara | 80 | 0 | 0 |
| aimon | 80 | 0 | 0 |
| llm_judge | 80 | 0 | 0 |
| ragtruth | 80 | 0 | 0 |

## Aggregated Metrics by Case Type

### vectara

| Case Type | Cases | RAG Wins | BASELINE Wins | Ties | RAG Factual | RAG Hallucinated | BASELINE Factual | BASELINE Hallucinated | Skipped | Errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| multi_hop | 80 | 34 | 10 | 36 | 63 | 17 | 35 | 45 | 0 | 0 |

### aimon

| Case Type | Cases | RAG Wins | BASELINE Wins | Ties | RAG Factual | RAG Hallucinated | BASELINE Factual | BASELINE Hallucinated | Skipped | Errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| multi_hop | 80 | 53 | 11 | 16 | 50 | 30 | 29 | 51 | 0 | 0 |

### llm_judge

| Case Type | Cases | RAG Wins | BASELINE Wins | Ties | RAG Factual | RAG Hallucinated | BASELINE Factual | BASELINE Hallucinated | Skipped | Errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| multi_hop | 80 | 41 | 5 | 34 | 62 | 18 | 32 | 48 | 0 | 0 |

### ragtruth

| Case Type | Cases | RAG Wins | BASELINE Wins | Ties | RAG Factual | RAG Hallucinated | BASELINE Factual | BASELINE Hallucinated | Skipped | Errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| multi_hop | 80 | 43 | 18 | 19 | 56 | 24 | 23 | 57 | 0 | 0 |


## vectara Results

| # | Case ID | Question | RAG | BASELINE | RAG Score | BASELINE Score | Winner | Status |
|---:|---|---|---|---|---:|---:|---|---|
| 1 | multi_hop_001 | What capital corresponds to the country where Frida Kahlo was born? | factual | factual | 1.000 | 1.000 | Tie | completed |
| 2 | multi_hop_002 | For The Great Gatsby, what is the capital of its author's citizenship country? | factual | factual | 1.000 | 1.000 | Tie | completed |
| 3 | multi_hop_003 | What is the capital of the nation where Microsoft's founder was born? | factual | hallucinated | 1.000 | 0.000 | RAG | completed |
| 4 | multi_hop_004 | Identify the continent of Karlovac Gymnasium's country for Nikola Tesla. | factual | factual | 1.000 | 1.000 | Tie | completed |
| 5 | multi_hop_005 | Which capital is tied to the country of birth of Mexico's head of state? | factual | hallucinated | 1.000 | 1.000 | RAG | completed |
| 6 | multi_hop_006 | Identify the capital of the nation associated with Congressional Gold Medal that George Washington received. | hallucinated | factual | 0.000 | 1.000 | BASELINE | completed |
| 7 | multi_hop_007 | Which capital belongs to the country of Ada Lovelace's birthplace? | factual | hallucinated | 1.000 | 0.000 | RAG | completed |
| 8 | multi_hop_008 | What capital is associated with the author's country in Don Quixote? | factual | hallucinated | 1.000 | 1.000 | RAG | completed |
| 9 | multi_hop_009 | What is the capital of the country where the founder of Siemens was born? | factual | hallucinated | 1.000 | 0.000 | RAG | completed |
| 10 | multi_hop_010 | Identify the continent of Science Faculty of Paris's country for Marie Curie. | factual | hallucinated | 1.000 | 0.000 | RAG | completed |
| 11 | multi_hop_011 | What is the capital of the country where the head of state of South Africa was born? | hallucinated | hallucinated | 0.000 | 1.000 | BASELINE | completed |
| 12 | multi_hop_012 | Alan Turing received Officer of the Order of the British Empire; what is the capital of the award's country? | factual | factual | 1.000 | 1.000 | Tie | completed |
| 13 | multi_hop_013 | The birthplace of George Washington is in which country's capital city? | hallucinated | hallucinated | 0.000 | 1.000 | BASELINE | completed |
| 14 | multi_hop_014 | For Don Quixote, what is the capital of its author's citizenship country? | hallucinated | hallucinated | 0.000 | 0.000 | Tie | completed |
| 15 | multi_hop_015 | What is the capital of the country where the founder of Meta was born? | factual | hallucinated | 1.000 | 0.000 | RAG | completed |
| 16 | multi_hop_016 | Identify the continent of Massachusetts Institute of Technology's country for Richard Feynman. | factual | factual | 1.000 | 1.000 | Tie | completed |
| 17 | multi_hop_017 | Name the capital city of the birth country of Brazil's head of state. | factual | hallucinated | 1.000 | 1.000 | RAG | completed |
| 18 | multi_hop_018 | Isaac Newton received Knight Bachelor; what is the capital of the award's country? | factual | factual | 1.000 | 1.000 | Tie | completed |
| 19 | multi_hop_019 | What is the capital of the country where Charles Darwin was born? | factual | factual | 1.000 | 1.000 | Tie | completed |
| 20 | multi_hop_020 | Identify the capital of the nation linked to the author of Moby-Dick. | factual | hallucinated | 1.000 | 1.000 | RAG | completed |
| 21 | multi_hop_021 | Which capital corresponds to the founder-birth country of Cisco? | factual | hallucinated | 1.000 | 1.000 | RAG | completed |
| 22 | multi_hop_022 | The country of Albert Einstein's alma mater, ETH Zurich, lies on what continent? | factual | factual | 1.000 | 1.000 | Tie | completed |
| 23 | multi_hop_023 | What is the capital of the country where the head of state of Australia was born? | factual | factual | 1.000 | 1.000 | Tie | completed |
| 24 | multi_hop_024 | Charles Darwin received Copley Medal; what is the capital of the award's country? | factual | factual | 1.000 | 1.000 | Tie | completed |
| 25 | multi_hop_025 | What capital corresponds to the country where Nelson Mandela was born? | hallucinated | hallucinated | 0.000 | 0.000 | Tie | completed |
| 26 | multi_hop_026 | Name the capital city of the author's country for The Brothers Karamazov. | factual | factual | 1.000 | 1.000 | Tie | completed |
| 27 | multi_hop_027 | Which capital corresponds to the founder-birth country of Samsung Electronics? | factual | hallucinated | 1.000 | 1.000 | RAG | completed |
| 28 | multi_hop_028 | Identify the continent of Charles University's country for Nikola Tesla. | factual | hallucinated | 1.000 | 1.000 | RAG | completed |
| 29 | multi_hop_029 | For United Kingdom, which capital belongs to its head of state's birth country? | factual | hallucinated | 1.000 | 1.000 | RAG | completed |
| 30 | multi_hop_030 | Name the capital city of the country linked to Grand Cross of the Order of the Sun of Peru, received by Angela Merkel. | factual | hallucinated | 1.000 | 1.000 | RAG | completed |
| 31 | multi_hop_031 | Which capital belongs to the country of Abraham Lincoln's birthplace? | factual | hallucinated | 1.000 | 0.000 | RAG | completed |
| 32 | multi_hop_032 | What is the capital of the nation where Intel's founder was born? | factual | hallucinated | 1.000 | 1.000 | RAG | completed |
| 33 | multi_hop_033 | Which continent includes the country where Stephen Hawking attended Trinity Hall? | factual | factual | 1.000 | 1.000 | Tie | completed |
| 34 | multi_hop_034 | The head of state of United States was born in a country with what capital? | hallucinated | hallucinated | 0.000 | 1.000 | BASELINE | completed |
| 35 | multi_hop_035 | What capital is associated with the country tied to Time Person of the Year received by Mahatma Gandhi? | factual | hallucinated | 0.250 | 0.333 | RAG | completed |
| 36 | multi_hop_036 | What is the capital of the country where Barack Obama was born? | hallucinated | hallucinated | 0.000 | 0.000 | Tie | completed |
| 37 | multi_hop_037 | Identify the capital of the country containing the founder's birthplace of Intel. | factual | hallucinated | 1.000 | 1.000 | RAG | completed |
| 38 | multi_hop_038 | On which continent is the country where Isaac Newton studied at University of Cambridge? | factual | factual | 1.000 | 1.000 | Tie | completed |
| 39 | multi_hop_039 | What is the capital of the country where the head of state of Spain was born? | factual | hallucinated | 1.000 | 1.000 | RAG | completed |
| 40 | multi_hop_040 | Which capital belongs to the country connected to Willard Gibbs Award received by Marie Curie? | factual | hallucinated | 1.000 | 1.000 | RAG | completed |
| 41 | multi_hop_041 | What is the capital of the country where Grace Hopper was born? | factual | factual | 1.000 | 1.000 | Tie | completed |
| 42 | multi_hop_042 | What is the capital of the country where the founder of Nvidia was born? | factual | hallucinated | 1.000 | 1.000 | RAG | completed |
| 43 | multi_hop_043 | Identify the continent of University of Cambridge's country for Niels Bohr. | factual | hallucinated | 1.000 | 1.000 | RAG | completed |
| 44 | multi_hop_044 | For Japan, which capital belongs to its head of state's birth country? | factual | hallucinated | 1.000 | 1.000 | RAG | completed |
| 45 | multi_hop_045 | What is the capital of the country associated with the award Angela Merkel received (Grand Cross of the Order of Prince Henry)? | factual | factual | 1.000 | 1.000 | Tie | completed |
| 46 | multi_hop_046 | Name the capital city of the nation in which Galileo Galilei was born. | hallucinated | factual | 0.000 | 1.000 | BASELINE | completed |
| 47 | multi_hop_047 | What is the capital of the nation where Oracle Corporation's founder was born? | factual | factual | 1.000 | 1.000 | Tie | completed |
| 48 | multi_hop_048 | Identify the continent of King's College's country for Alan Turing. | factual | factual | 1.000 | 1.000 | Tie | completed |
| 49 | multi_hop_049 | For France, which capital belongs to its head of state's birth country? | factual | hallucinated | 1.000 | 1.000 | RAG | completed |
| 50 | multi_hop_050 | For Charles Darwin and the award Pour le Mérite for Sciences and Arts order, what is the country's capital? | hallucinated | hallucinated | 1.000 | 0.000 | RAG | completed |
| 51 | multi_hop_051 | Identify the capital of the country containing Winston Churchill's birth location. | factual | factual | 1.000 | 1.000 | Tie | completed |
| 52 | multi_hop_052 | What is the capital of the country where the founder of Tesla, Inc. was born? | factual | hallucinated | 1.000 | 0.000 | RAG | completed |
| 53 | multi_hop_053 | Name the continent of the country that contains University of Paris, where Marie Curie was educated. | factual | factual | 1.000 | 1.000 | Tie | completed |
| 54 | multi_hop_054 | What capital corresponds to the birth-country of the current head of state of Netherlands? | hallucinated | factual | 0.000 | 1.000 | BASELINE | completed |
| 55 | multi_hop_055 | For Albert Einstein and the award Barnard Medal for Meritorious Service to Science, what is the country's capital? | factual | hallucinated | 1.000 | 1.000 | RAG | completed |
| 56 | multi_hop_056 | The birthplace of Alan Turing is in which country's capital city? | factual | factual | 1.000 | 1.000 | Tie | completed |
| 57 | multi_hop_057 | Name the capital city of the founder's birth country for Meta. | factual | hallucinated | 1.000 | 1.000 | RAG | completed |
| 58 | multi_hop_058 | Stephen Hawking studied at St Albans School in a country on which continent? | factual | factual | 1.000 | 1.000 | Tie | completed |
| 59 | multi_hop_059 | For Canada, which capital belongs to its head of state's birth country? | factual | hallucinated | 1.000 | 1.000 | RAG | completed |
| 60 | multi_hop_060 | What is the capital of the country associated with the award Albert Einstein received (Nobel Prize in Physics)? | factual | factual | 1.000 | 1.000 | Tie | completed |
| 61 | multi_hop_061 | What capital corresponds to the country where Mahatma Gandhi was born? | factual | factual | 1.000 | 1.000 | Tie | completed |
| 62 | multi_hop_062 | What is the capital of the nation where Sony Group's founder was born? | factual | factual | 1.000 | 1.000 | Tie | completed |
| 63 | multi_hop_063 | Which continent includes the country where Marie Curie attended Flying University? | hallucinated | factual | 1.000 | 1.000 | BASELINE | completed |
| 64 | multi_hop_064 | For India, which capital belongs to its head of state's birth country? | factual | hallucinated | 1.000 | 1.000 | RAG | completed |
| 65 | multi_hop_065 | What capital is associated with the country tied to Fellow of the Royal Society received by Winston Churchill? | hallucinated | hallucinated | 1.000 | 1.000 | Tie | completed |
| 66 | multi_hop_066 | Marie Curie was born in a country whose capital is what? | factual | factual | 0.250 | 0.250 | Tie | completed |
| 67 | multi_hop_067 | What is the capital of the nation where Cisco's founder was born? | factual | hallucinated | 1.000 | 1.000 | RAG | completed |
| 68 | multi_hop_068 | Name the continent of the country that contains Newnham College, where Rosalind Franklin was educated. | factual | factual | 1.000 | 1.000 | Tie | completed |
| 69 | multi_hop_069 | For Italy, which capital belongs to its head of state's birth country? | hallucinated | hallucinated | 0.000 | 1.000 | BASELINE | completed |
| 70 | multi_hop_070 | For Martin Luther King Jr. and the award Jawaharlal Nehru Award for International Understanding, what is the country's capital? | hallucinated | hallucinated | 0.000 | 0.000 | Tie | completed |
| 71 | multi_hop_071 | Identify the capital of the country containing Martin Luther King Jr.'s birth location. | factual | hallucinated | 1.000 | 0.000 | RAG | completed |
| 72 | multi_hop_072 | What is the capital of the country where the founder of Intel was born? | hallucinated | hallucinated | 0.000 | 0.000 | Tie | completed |
| 73 | multi_hop_073 | Richard Feynman studied at Princeton University in a country on which continent? | factual | factual | 1.000 | 1.000 | Tie | completed |
| 74 | multi_hop_074 | What is the capital of the country where the head of state of South Africa was born? | hallucinated | hallucinated | 0.000 | 1.000 | BASELINE | completed |
| 75 | multi_hop_075 | What capital is associated with the country tied to Gold Medal of the Royal Astronomical Society received by Albert Einstein? | factual | hallucinated | 1.000 | 0.000 | RAG | completed |
| 76 | multi_hop_076 | What capital corresponds to the country where Isaac Newton was born? | factual | factual | 1.000 | 1.000 | Tie | completed |
| 77 | multi_hop_077 | The founder of Google was born in a country whose capital is what? | factual | factual | 1.000 | 1.000 | Tie | completed |
| 78 | multi_hop_078 | The country of Stephen Hawking's alma mater, University College, Oxford, lies on what continent? | factual | factual | 1.000 | 1.000 | Tie | completed |
| 79 | multi_hop_079 | Which capital is tied to the country of birth of Germany's head of state? | factual | hallucinated | 1.000 | 1.000 | RAG | completed |
| 80 | multi_hop_080 | Identify the capital of the nation associated with Nobel Prize in Chemistry that Marie Curie received. | hallucinated | factual | 0.000 | 1.000 | BASELINE | completed |

## aimon Results

| # | Case ID | Question | RAG | BASELINE | RAG Score | BASELINE Score | Winner | Status |
|---:|---|---|---|---|---:|---:|---|---|
| 1 | multi_hop_001 | What capital corresponds to the country where Frida Kahlo was born? | factual | factual | 0.445 | 0.445 | Tie | completed |
| 2 | multi_hop_002 | For The Great Gatsby, what is the capital of its author's citizenship country? | factual | factual | 0.045 | 0.040 | Tie | completed |
| 3 | multi_hop_003 | What is the capital of the nation where Microsoft's founder was born? | factual | hallucinated | 0.056 | 0.820 | RAG | completed |
| 4 | multi_hop_004 | Identify the continent of Karlovac Gymnasium's country for Nikola Tesla. | hallucinated | factual | 0.891 | 0.073 | BASELINE | completed |
| 5 | multi_hop_005 | Which capital is tied to the country of birth of Mexico's head of state? | factual | hallucinated | 0.049 | 0.992 | RAG | completed |
| 6 | multi_hop_006 | Identify the capital of the nation associated with Congressional Gold Medal that George Washington received. | hallucinated | factual | 0.895 | 0.091 | BASELINE | completed |
| 7 | multi_hop_007 | Which capital belongs to the country of Ada Lovelace's birthplace? | factual | hallucinated | 0.130 | 0.973 | RAG | completed |
| 8 | multi_hop_008 | What capital is associated with the author's country in Don Quixote? | factual | hallucinated | 0.005 | 0.996 | RAG | completed |
| 9 | multi_hop_009 | What is the capital of the country where the founder of Siemens was born? | factual | hallucinated | 0.116 | 0.914 | RAG | completed |
| 10 | multi_hop_010 | Identify the continent of Science Faculty of Paris's country for Marie Curie. | factual | hallucinated | 0.241 | 0.980 | RAG | completed |
| 11 | multi_hop_011 | What is the capital of the country where the head of state of South Africa was born? | hallucinated | hallucinated | 0.961 | 0.969 | Tie | completed |
| 12 | multi_hop_012 | Alan Turing received Officer of the Order of the British Empire; what is the capital of the award's country? | factual | hallucinated | 0.115 | 0.773 | RAG | completed |
| 13 | multi_hop_013 | The birthplace of George Washington is in which country's capital city? | hallucinated | hallucinated | 0.914 | 0.984 | RAG | completed |
| 14 | multi_hop_014 | For Don Quixote, what is the capital of its author's citizenship country? | hallucinated | hallucinated | 0.805 | 0.828 | Tie | completed |
| 15 | multi_hop_015 | What is the capital of the country where the founder of Meta was born? | factual | hallucinated | 0.097 | 0.914 | RAG | completed |
| 16 | multi_hop_016 | Identify the continent of Massachusetts Institute of Technology's country for Richard Feynman. | factual | factual | 0.046 | 0.195 | RAG | completed |
| 17 | multi_hop_017 | Name the capital city of the birth country of Brazil's head of state. | hallucinated | hallucinated | 0.652 | 0.992 | RAG | completed |
| 18 | multi_hop_018 | Isaac Newton received Knight Bachelor; what is the capital of the award's country? | factual | factual | 0.108 | 0.130 | Tie | completed |
| 19 | multi_hop_019 | What is the capital of the country where Charles Darwin was born? | factual | factual | 0.169 | 0.169 | Tie | completed |
| 20 | multi_hop_020 | Identify the capital of the nation linked to the author of Moby-Dick. | factual | hallucinated | 0.104 | 0.980 | RAG | completed |
| 21 | multi_hop_021 | Which capital corresponds to the founder-birth country of Cisco? | hallucinated | hallucinated | 0.578 | 0.980 | RAG | completed |
| 22 | multi_hop_022 | The country of Albert Einstein's alma mater, ETH Zurich, lies on what continent? | factual | factual | 0.012 | 0.058 | Tie | completed |
| 23 | multi_hop_023 | What is the capital of the country where the head of state of Australia was born? | factual | factual | 0.469 | 0.441 | Tie | completed |
| 24 | multi_hop_024 | Charles Darwin received Copley Medal; what is the capital of the award's country? | factual | hallucinated | 0.098 | 0.598 | RAG | completed |
| 25 | multi_hop_025 | What capital corresponds to the country where Nelson Mandela was born? | hallucinated | hallucinated | 0.957 | 0.957 | Tie | completed |
| 26 | multi_hop_026 | Name the capital city of the author's country for The Brothers Karamazov. | factual | factual | 0.234 | 0.299 | RAG | completed |
| 27 | multi_hop_027 | Which capital corresponds to the founder-birth country of Samsung Electronics? | factual | hallucinated | 0.081 | 0.992 | RAG | completed |
| 28 | multi_hop_028 | Identify the continent of Charles University's country for Nikola Tesla. | factual | hallucinated | 0.069 | 0.977 | RAG | completed |
| 29 | multi_hop_029 | For United Kingdom, which capital belongs to its head of state's birth country? | factual | hallucinated | 0.063 | 0.941 | RAG | completed |
| 30 | multi_hop_030 | Name the capital city of the country linked to Grand Cross of the Order of the Sun of Peru, received by Angela Merkel. | factual | hallucinated | 0.346 | 0.602 | RAG | completed |
| 31 | multi_hop_031 | Which capital belongs to the country of Abraham Lincoln's birthplace? | factual | hallucinated | 0.087 | 0.953 | RAG | completed |
| 32 | multi_hop_032 | What is the capital of the nation where Intel's founder was born? | factual | hallucinated | 0.048 | 0.977 | RAG | completed |
| 33 | multi_hop_033 | Which continent includes the country where Stephen Hawking attended Trinity Hall? | factual | factual | 0.012 | 0.103 | RAG | completed |
| 34 | multi_hop_034 | The head of state of United States was born in a country with what capital? | hallucinated | hallucinated | 0.938 | 0.992 | RAG | completed |
| 35 | multi_hop_035 | What capital is associated with the country tied to Time Person of the Year received by Mahatma Gandhi? | hallucinated | hallucinated | 0.898 | 0.984 | RAG | completed |
| 36 | multi_hop_036 | What is the capital of the country where Barack Obama was born? | hallucinated | hallucinated | 0.910 | 0.910 | Tie | completed |
| 37 | multi_hop_037 | Identify the capital of the country containing the founder's birthplace of Intel. | factual | hallucinated | 0.125 | 0.980 | RAG | completed |
| 38 | multi_hop_038 | On which continent is the country where Isaac Newton studied at University of Cambridge? | factual | factual | 0.002 | 0.318 | RAG | completed |
| 39 | multi_hop_039 | What is the capital of the country where the head of state of Spain was born? | factual | hallucinated | 0.050 | 0.996 | RAG | completed |
| 40 | multi_hop_040 | Which capital belongs to the country connected to Willard Gibbs Award received by Marie Curie? | factual | hallucinated | 0.025 | 0.695 | RAG | completed |
| 41 | multi_hop_041 | What is the capital of the country where Grace Hopper was born? | factual | factual | 0.076 | 0.013 | BASELINE | completed |
| 42 | multi_hop_042 | What is the capital of the country where the founder of Nvidia was born? | factual | hallucinated | 0.138 | 0.992 | RAG | completed |
| 43 | multi_hop_043 | Identify the continent of University of Cambridge's country for Niels Bohr. | factual | hallucinated | 0.046 | 0.969 | RAG | completed |
| 44 | multi_hop_044 | For Japan, which capital belongs to its head of state's birth country? | hallucinated | hallucinated | 0.535 | 0.980 | RAG | completed |
| 45 | multi_hop_045 | What is the capital of the country associated with the award Angela Merkel received (Grand Cross of the Order of Prince Henry)? | factual | factual | 0.168 | 0.297 | RAG | completed |
| 46 | multi_hop_046 | Name the capital city of the nation in which Galileo Galilei was born. | hallucinated | factual | 0.898 | 0.169 | BASELINE | completed |
| 47 | multi_hop_047 | What is the capital of the nation where Oracle Corporation's founder was born? | factual | factual | 0.118 | 0.436 | RAG | completed |
| 48 | multi_hop_048 | Identify the continent of King's College's country for Alan Turing. | factual | factual | 0.020 | 0.320 | RAG | completed |
| 49 | multi_hop_049 | For France, which capital belongs to its head of state's birth country? | hallucinated | hallucinated | 0.863 | 0.953 | RAG | completed |
| 50 | multi_hop_050 | For Charles Darwin and the award Pour le Mérite for Sciences and Arts order, what is the country's capital? | hallucinated | hallucinated | 0.949 | 0.949 | Tie | completed |
| 51 | multi_hop_051 | Identify the capital of the country containing Winston Churchill's birth location. | hallucinated | hallucinated | 0.699 | 0.949 | RAG | completed |
| 52 | multi_hop_052 | What is the capital of the country where the founder of Tesla, Inc. was born? | factual | hallucinated | 0.213 | 0.973 | RAG | completed |
| 53 | multi_hop_053 | Name the continent of the country that contains University of Paris, where Marie Curie was educated. | hallucinated | factual | 0.512 | 0.025 | BASELINE | completed |
| 54 | multi_hop_054 | What capital corresponds to the birth-country of the current head of state of Netherlands? | hallucinated | factual | 0.934 | 0.293 | BASELINE | completed |
| 55 | multi_hop_055 | For Albert Einstein and the award Barnard Medal for Meritorious Service to Science, what is the country's capital? | factual | hallucinated | 0.202 | 0.984 | RAG | completed |
| 56 | multi_hop_056 | The birthplace of Alan Turing is in which country's capital city? | factual | factual | 0.073 | 0.156 | RAG | completed |
| 57 | multi_hop_057 | Name the capital city of the founder's birth country for Meta. | factual | hallucinated | 0.400 | 0.977 | RAG | completed |
| 58 | multi_hop_058 | Stephen Hawking studied at St Albans School in a country on which continent? | factual | factual | 0.158 | 0.245 | RAG | completed |
| 59 | multi_hop_059 | For Canada, which capital belongs to its head of state's birth country? | factual | hallucinated | 0.157 | 0.965 | RAG | completed |
| 60 | multi_hop_060 | What is the capital of the country associated with the award Albert Einstein received (Nobel Prize in Physics)? | factual | factual | 0.247 | 0.432 | RAG | completed |
| 61 | multi_hop_061 | What capital corresponds to the country where Mahatma Gandhi was born? | hallucinated | hallucinated | 0.812 | 0.934 | RAG | completed |
| 62 | multi_hop_062 | What is the capital of the nation where Sony Group's founder was born? | hallucinated | hallucinated | 0.684 | 0.684 | Tie | completed |
| 63 | multi_hop_063 | Which continent includes the country where Marie Curie attended Flying University? | hallucinated | factual | 0.957 | 0.221 | BASELINE | completed |
| 64 | multi_hop_064 | For India, which capital belongs to its head of state's birth country? | hallucinated | hallucinated | 0.875 | 0.957 | RAG | completed |
| 65 | multi_hop_065 | What capital is associated with the country tied to Fellow of the Royal Society received by Winston Churchill? | hallucinated | hallucinated | 0.973 | 0.973 | Tie | completed |
| 66 | multi_hop_066 | Marie Curie was born in a country whose capital is what? | hallucinated | hallucinated | 0.617 | 0.605 | Tie | completed |
| 67 | multi_hop_067 | What is the capital of the nation where Cisco's founder was born? | factual | hallucinated | 0.067 | 0.984 | RAG | completed |
| 68 | multi_hop_068 | Name the continent of the country that contains Newnham College, where Rosalind Franklin was educated. | factual | factual | 0.010 | 0.103 | RAG | completed |
| 69 | multi_hop_069 | For Italy, which capital belongs to its head of state's birth country? | hallucinated | hallucinated | 0.684 | 0.961 | RAG | completed |
| 70 | multi_hop_070 | For Martin Luther King Jr. and the award Jawaharlal Nehru Award for International Understanding, what is the country's capital? | hallucinated | hallucinated | 0.965 | 0.898 | BASELINE | completed |
| 71 | multi_hop_071 | Identify the capital of the country containing Martin Luther King Jr.'s birth location. | factual | hallucinated | 0.108 | 0.840 | RAG | completed |
| 72 | multi_hop_072 | What is the capital of the country where the founder of Intel was born? | hallucinated | hallucinated | 0.918 | 0.980 | RAG | completed |
| 73 | multi_hop_073 | Richard Feynman studied at Princeton University in a country on which continent? | factual | factual | 0.188 | 0.036 | BASELINE | completed |
| 74 | multi_hop_074 | What is the capital of the country where the head of state of South Africa was born? | hallucinated | hallucinated | 0.980 | 0.996 | Tie | completed |
| 75 | multi_hop_075 | What capital is associated with the country tied to Gold Medal of the Royal Astronomical Society received by Albert Einstein? | factual | hallucinated | 0.136 | 0.973 | RAG | completed |
| 76 | multi_hop_076 | What capital corresponds to the country where Isaac Newton was born? | factual | factual | 0.183 | 0.183 | Tie | completed |
| 77 | multi_hop_077 | The founder of Google was born in a country whose capital is what? | factual | factual | 0.355 | 0.215 | BASELINE | completed |
| 78 | multi_hop_078 | The country of Stephen Hawking's alma mater, University College, Oxford, lies on what continent? | factual | factual | 0.012 | 0.119 | RAG | completed |
| 79 | multi_hop_079 | Which capital is tied to the country of birth of Germany's head of state? | hallucinated | hallucinated | 0.848 | 0.949 | RAG | completed |
| 80 | multi_hop_080 | Identify the capital of the nation associated with Nobel Prize in Chemistry that Marie Curie received. | hallucinated | factual | 0.867 | 0.105 | BASELINE | completed |

## llm_judge Results

| # | Case ID | Question | RAG | BASELINE | RAG Score | BASELINE Score | Winner | Status |
|---:|---|---|---|---|---:|---:|---|---|
| 1 | multi_hop_001 | What capital corresponds to the country where Frida Kahlo was born? | factual | factual | N/A | N/A | Tie | completed |
| 2 | multi_hop_002 | For The Great Gatsby, what is the capital of its author's citizenship country? | factual | factual | N/A | N/A | Tie | completed |
| 3 | multi_hop_003 | What is the capital of the nation where Microsoft's founder was born? | factual | hallucinated | N/A | N/A | RAG | completed |
| 4 | multi_hop_004 | Identify the continent of Karlovac Gymnasium's country for Nikola Tesla. | hallucinated | factual | N/A | N/A | BASELINE | completed |
| 5 | multi_hop_005 | Which capital is tied to the country of birth of Mexico's head of state? | factual | hallucinated | N/A | N/A | RAG | completed |
| 6 | multi_hop_006 | Identify the capital of the nation associated with Congressional Gold Medal that George Washington received. | hallucinated | factual | N/A | N/A | BASELINE | completed |
| 7 | multi_hop_007 | Which capital belongs to the country of Ada Lovelace's birthplace? | factual | hallucinated | N/A | N/A | RAG | completed |
| 8 | multi_hop_008 | What capital is associated with the author's country in Don Quixote? | factual | hallucinated | N/A | N/A | RAG | completed |
| 9 | multi_hop_009 | What is the capital of the country where the founder of Siemens was born? | factual | hallucinated | N/A | N/A | RAG | completed |
| 10 | multi_hop_010 | Identify the continent of Science Faculty of Paris's country for Marie Curie. | factual | hallucinated | N/A | N/A | RAG | completed |
| 11 | multi_hop_011 | What is the capital of the country where the head of state of South Africa was born? | hallucinated | hallucinated | N/A | N/A | Tie | completed |
| 12 | multi_hop_012 | Alan Turing received Officer of the Order of the British Empire; what is the capital of the award's country? | factual | factual | N/A | N/A | RAG | completed |
| 13 | multi_hop_013 | The birthplace of George Washington is in which country's capital city? | hallucinated | hallucinated | N/A | N/A | Tie | completed |
| 14 | multi_hop_014 | For Don Quixote, what is the capital of its author's citizenship country? | hallucinated | hallucinated | N/A | N/A | Tie | completed |
| 15 | multi_hop_015 | What is the capital of the country where the founder of Meta was born? | factual | hallucinated | N/A | N/A | RAG | completed |
| 16 | multi_hop_016 | Identify the continent of Massachusetts Institute of Technology's country for Richard Feynman. | factual | factual | N/A | N/A | Tie | completed |
| 17 | multi_hop_017 | Name the capital city of the birth country of Brazil's head of state. | factual | hallucinated | N/A | N/A | RAG | completed |
| 18 | multi_hop_018 | Isaac Newton received Knight Bachelor; what is the capital of the award's country? | factual | factual | N/A | N/A | Tie | completed |
| 19 | multi_hop_019 | What is the capital of the country where Charles Darwin was born? | factual | factual | N/A | N/A | Tie | completed |
| 20 | multi_hop_020 | Identify the capital of the nation linked to the author of Moby-Dick. | factual | hallucinated | N/A | N/A | RAG | completed |
| 21 | multi_hop_021 | Which capital corresponds to the founder-birth country of Cisco? | factual | hallucinated | N/A | N/A | RAG | completed |
| 22 | multi_hop_022 | The country of Albert Einstein's alma mater, ETH Zurich, lies on what continent? | factual | factual | N/A | N/A | Tie | completed |
| 23 | multi_hop_023 | What is the capital of the country where the head of state of Australia was born? | factual | factual | N/A | N/A | RAG | completed |
| 24 | multi_hop_024 | Charles Darwin received Copley Medal; what is the capital of the award's country? | factual | factual | N/A | N/A | Tie | completed |
| 25 | multi_hop_025 | What capital corresponds to the country where Nelson Mandela was born? | hallucinated | hallucinated | N/A | N/A | Tie | completed |
| 26 | multi_hop_026 | Name the capital city of the author's country for The Brothers Karamazov. | factual | factual | N/A | N/A | Tie | completed |
| 27 | multi_hop_027 | Which capital corresponds to the founder-birth country of Samsung Electronics? | factual | hallucinated | N/A | N/A | RAG | completed |
| 28 | multi_hop_028 | Identify the continent of Charles University's country for Nikola Tesla. | factual | hallucinated | N/A | N/A | RAG | completed |
| 29 | multi_hop_029 | For United Kingdom, which capital belongs to its head of state's birth country? | factual | hallucinated | N/A | N/A | RAG | completed |
| 30 | multi_hop_030 | Name the capital city of the country linked to Grand Cross of the Order of the Sun of Peru, received by Angela Merkel. | factual | hallucinated | N/A | N/A | RAG | completed |
| 31 | multi_hop_031 | Which capital belongs to the country of Abraham Lincoln's birthplace? | factual | hallucinated | N/A | N/A | RAG | completed |
| 32 | multi_hop_032 | What is the capital of the nation where Intel's founder was born? | factual | hallucinated | N/A | N/A | RAG | completed |
| 33 | multi_hop_033 | Which continent includes the country where Stephen Hawking attended Trinity Hall? | factual | factual | N/A | N/A | RAG | completed |
| 34 | multi_hop_034 | The head of state of United States was born in a country with what capital? | hallucinated | hallucinated | N/A | N/A | Tie | completed |
| 35 | multi_hop_035 | What capital is associated with the country tied to Time Person of the Year received by Mahatma Gandhi? | hallucinated | hallucinated | N/A | N/A | Tie | completed |
| 36 | multi_hop_036 | What is the capital of the country where Barack Obama was born? | hallucinated | hallucinated | N/A | N/A | Tie | completed |
| 37 | multi_hop_037 | Identify the capital of the country containing the founder's birthplace of Intel. | factual | hallucinated | N/A | N/A | RAG | completed |
| 38 | multi_hop_038 | On which continent is the country where Isaac Newton studied at University of Cambridge? | factual | factual | N/A | N/A | RAG | completed |
| 39 | multi_hop_039 | What is the capital of the country where the head of state of Spain was born? | factual | hallucinated | N/A | N/A | RAG | completed |
| 40 | multi_hop_040 | Which capital belongs to the country connected to Willard Gibbs Award received by Marie Curie? | factual | hallucinated | N/A | N/A | RAG | completed |
| 41 | multi_hop_041 | What is the capital of the country where Grace Hopper was born? | factual | factual | N/A | N/A | Tie | completed |
| 42 | multi_hop_042 | What is the capital of the country where the founder of Nvidia was born? | factual | hallucinated | N/A | N/A | RAG | completed |
| 43 | multi_hop_043 | Identify the continent of University of Cambridge's country for Niels Bohr. | factual | hallucinated | N/A | N/A | RAG | completed |
| 44 | multi_hop_044 | For Japan, which capital belongs to its head of state's birth country? | factual | hallucinated | N/A | N/A | RAG | completed |
| 45 | multi_hop_045 | What is the capital of the country associated with the award Angela Merkel received (Grand Cross of the Order of Prince Henry)? | factual | factual | N/A | N/A | Tie | completed |
| 46 | multi_hop_046 | Name the capital city of the nation in which Galileo Galilei was born. | factual | factual | N/A | N/A | Tie | completed |
| 47 | multi_hop_047 | What is the capital of the nation where Oracle Corporation's founder was born? | factual | factual | N/A | N/A | Tie | completed |
| 48 | multi_hop_048 | Identify the continent of King's College's country for Alan Turing. | factual | factual | N/A | N/A | RAG | completed |
| 49 | multi_hop_049 | For France, which capital belongs to its head of state's birth country? | factual | hallucinated | N/A | N/A | RAG | completed |
| 50 | multi_hop_050 | For Charles Darwin and the award Pour le Mérite for Sciences and Arts order, what is the country's capital? | hallucinated | hallucinated | N/A | N/A | Tie | completed |
| 51 | multi_hop_051 | Identify the capital of the country containing Winston Churchill's birth location. | factual | hallucinated | N/A | N/A | RAG | completed |
| 52 | multi_hop_052 | What is the capital of the country where the founder of Tesla, Inc. was born? | factual | hallucinated | N/A | N/A | RAG | completed |
| 53 | multi_hop_053 | Name the continent of the country that contains University of Paris, where Marie Curie was educated. | factual | factual | N/A | N/A | Tie | completed |
| 54 | multi_hop_054 | What capital corresponds to the birth-country of the current head of state of Netherlands? | hallucinated | factual | N/A | N/A | BASELINE | completed |
| 55 | multi_hop_055 | For Albert Einstein and the award Barnard Medal for Meritorious Service to Science, what is the country's capital? | factual | hallucinated | N/A | N/A | RAG | completed |
| 56 | multi_hop_056 | The birthplace of Alan Turing is in which country's capital city? | factual | factual | N/A | N/A | RAG | completed |
| 57 | multi_hop_057 | Name the capital city of the founder's birth country for Meta. | factual | hallucinated | N/A | N/A | RAG | completed |
| 58 | multi_hop_058 | Stephen Hawking studied at St Albans School in a country on which continent? | factual | factual | N/A | N/A | Tie | completed |
| 59 | multi_hop_059 | For Canada, which capital belongs to its head of state's birth country? | factual | hallucinated | N/A | N/A | RAG | completed |
| 60 | multi_hop_060 | What is the capital of the country associated with the award Albert Einstein received (Nobel Prize in Physics)? | factual | factual | N/A | N/A | Tie | completed |
| 61 | multi_hop_061 | What capital corresponds to the country where Mahatma Gandhi was born? | factual | hallucinated | N/A | N/A | RAG | completed |
| 62 | multi_hop_062 | What is the capital of the nation where Sony Group's founder was born? | factual | factual | N/A | N/A | Tie | completed |
| 63 | multi_hop_063 | Which continent includes the country where Marie Curie attended Flying University? | hallucinated | factual | N/A | N/A | BASELINE | completed |
| 64 | multi_hop_064 | For India, which capital belongs to its head of state's birth country? | factual | hallucinated | N/A | N/A | RAG | completed |
| 65 | multi_hop_065 | What capital is associated with the country tied to Fellow of the Royal Society received by Winston Churchill? | hallucinated | hallucinated | N/A | N/A | Tie | completed |
| 66 | multi_hop_066 | Marie Curie was born in a country whose capital is what? | hallucinated | hallucinated | N/A | N/A | Tie | completed |
| 67 | multi_hop_067 | What is the capital of the nation where Cisco's founder was born? | factual | hallucinated | N/A | N/A | RAG | completed |
| 68 | multi_hop_068 | Name the continent of the country that contains Newnham College, where Rosalind Franklin was educated. | factual | factual | N/A | N/A | Tie | completed |
| 69 | multi_hop_069 | For Italy, which capital belongs to its head of state's birth country? | hallucinated | hallucinated | N/A | N/A | Tie | completed |
| 70 | multi_hop_070 | For Martin Luther King Jr. and the award Jawaharlal Nehru Award for International Understanding, what is the country's capital? | hallucinated | hallucinated | N/A | N/A | Tie | completed |
| 71 | multi_hop_071 | Identify the capital of the country containing Martin Luther King Jr.'s birth location. | factual | hallucinated | N/A | N/A | RAG | completed |
| 72 | multi_hop_072 | What is the capital of the country where the founder of Intel was born? | hallucinated | hallucinated | N/A | N/A | Tie | completed |
| 73 | multi_hop_073 | Richard Feynman studied at Princeton University in a country on which continent? | factual | factual | N/A | N/A | Tie | completed |
| 74 | multi_hop_074 | What is the capital of the country where the head of state of South Africa was born? | factual | hallucinated | N/A | N/A | RAG | completed |
| 75 | multi_hop_075 | What capital is associated with the country tied to Gold Medal of the Royal Astronomical Society received by Albert Einstein? | factual | hallucinated | N/A | N/A | RAG | completed |
| 76 | multi_hop_076 | What capital corresponds to the country where Isaac Newton was born? | factual | factual | N/A | N/A | Tie | completed |
| 77 | multi_hop_077 | The founder of Google was born in a country whose capital is what? | factual | factual | N/A | N/A | Tie | completed |
| 78 | multi_hop_078 | The country of Stephen Hawking's alma mater, University College, Oxford, lies on what continent? | factual | factual | N/A | N/A | Tie | completed |
| 79 | multi_hop_079 | Which capital is tied to the country of birth of Germany's head of state? | factual | hallucinated | N/A | N/A | RAG | completed |
| 80 | multi_hop_080 | Identify the capital of the nation associated with Nobel Prize in Chemistry that Marie Curie received. | hallucinated | factual | N/A | N/A | BASELINE | completed |

## ragtruth Results

| # | Case ID | Question | RAG | BASELINE | RAG Score | BASELINE Score | Winner | Status |
|---:|---|---|---|---|---:|---:|---|---|
| 1 | multi_hop_001 | What capital corresponds to the country where Frida Kahlo was born? | factual | factual | 0.000 | 0.000 | Tie | completed |
| 2 | multi_hop_002 | For The Great Gatsby, what is the capital of its author's citizenship country? | factual | factual | 0.000 | 0.000 | Tie | completed |
| 3 | multi_hop_003 | What is the capital of the nation where Microsoft's founder was born? | factual | hallucinated | 0.000 | 0.084 | RAG | completed |
| 4 | multi_hop_004 | Identify the continent of Karlovac Gymnasium's country for Nikola Tesla. | hallucinated | factual | 0.836 | 0.000 | BASELINE | completed |
| 5 | multi_hop_005 | Which capital is tied to the country of birth of Mexico's head of state? | factual | hallucinated | 0.000 | 0.000 | RAG | completed |
| 6 | multi_hop_006 | Identify the capital of the nation associated with Congressional Gold Medal that George Washington received. | hallucinated | factual | 0.101 | 0.000 | BASELINE | completed |
| 7 | multi_hop_007 | Which capital belongs to the country of Ada Lovelace's birthplace? | factual | hallucinated | 0.000 | 0.000 | RAG | completed |
| 8 | multi_hop_008 | What capital is associated with the author's country in Don Quixote? | factual | hallucinated | 0.000 | 0.000 | RAG | completed |
| 9 | multi_hop_009 | What is the capital of the country where the founder of Siemens was born? | factual | hallucinated | 0.000 | 0.655 | RAG | completed |
| 10 | multi_hop_010 | Identify the continent of Science Faculty of Paris's country for Marie Curie. | hallucinated | hallucinated | 0.845 | 0.000 | BASELINE | completed |
| 11 | multi_hop_011 | What is the capital of the country where the head of state of South Africa was born? | hallucinated | hallucinated | 0.176 | 0.000 | BASELINE | completed |
| 12 | multi_hop_012 | Alan Turing received Officer of the Order of the British Empire; what is the capital of the award's country? | factual | factual | 0.000 | 0.000 | Tie | completed |
| 13 | multi_hop_013 | The birthplace of George Washington is in which country's capital city? | hallucinated | hallucinated | 0.629 | 0.278 | BASELINE | completed |
| 14 | multi_hop_014 | For Don Quixote, what is the capital of its author's citizenship country? | hallucinated | hallucinated | 0.067 | 1.000 | RAG | completed |
| 15 | multi_hop_015 | What is the capital of the country where the founder of Meta was born? | factual | hallucinated | 0.000 | 0.583 | RAG | completed |
| 16 | multi_hop_016 | Identify the continent of Massachusetts Institute of Technology's country for Richard Feynman. | factual | factual | 0.000 | 0.000 | Tie | completed |
| 17 | multi_hop_017 | Name the capital city of the birth country of Brazil's head of state. | factual | hallucinated | 0.000 | 0.000 | RAG | completed |
| 18 | multi_hop_018 | Isaac Newton received Knight Bachelor; what is the capital of the award's country? | factual | hallucinated | 0.000 | 0.472 | RAG | completed |
| 19 | multi_hop_019 | What is the capital of the country where Charles Darwin was born? | factual | factual | 0.000 | 0.000 | Tie | completed |
| 20 | multi_hop_020 | Identify the capital of the nation linked to the author of Moby-Dick. | factual | hallucinated | 0.000 | 0.000 | RAG | completed |
| 21 | multi_hop_021 | Which capital corresponds to the founder-birth country of Cisco? | factual | hallucinated | 0.000 | 0.000 | RAG | completed |
| 22 | multi_hop_022 | The country of Albert Einstein's alma mater, ETH Zurich, lies on what continent? | factual | factual | 0.000 | 0.000 | Tie | completed |
| 23 | multi_hop_023 | What is the capital of the country where the head of state of Australia was born? | factual | hallucinated | 0.000 | 0.213 | RAG | completed |
| 24 | multi_hop_024 | Charles Darwin received Copley Medal; what is the capital of the award's country? | hallucinated | hallucinated | 0.412 | 0.972 | RAG | completed |
| 25 | multi_hop_025 | What capital corresponds to the country where Nelson Mandela was born? | hallucinated | hallucinated | 0.093 | 0.093 | Tie | completed |
| 26 | multi_hop_026 | Name the capital city of the author's country for The Brothers Karamazov. | hallucinated | factual | 0.250 | 0.000 | BASELINE | completed |
| 27 | multi_hop_027 | Which capital corresponds to the founder-birth country of Samsung Electronics? | factual | hallucinated | 0.000 | 0.000 | RAG | completed |
| 28 | multi_hop_028 | Identify the continent of Charles University's country for Nikola Tesla. | factual | hallucinated | 0.000 | 0.000 | RAG | completed |
| 29 | multi_hop_029 | For United Kingdom, which capital belongs to its head of state's birth country? | factual | hallucinated | 0.000 | 0.000 | RAG | completed |
| 30 | multi_hop_030 | Name the capital city of the country linked to Grand Cross of the Order of the Sun of Peru, received by Angela Merkel. | factual | hallucinated | 0.000 | 0.000 | RAG | completed |
| 31 | multi_hop_031 | Which capital belongs to the country of Abraham Lincoln's birthplace? | factual | hallucinated | 0.000 | 0.896 | RAG | completed |
| 32 | multi_hop_032 | What is the capital of the nation where Intel's founder was born? | factual | hallucinated | 0.000 | 0.000 | RAG | completed |
| 33 | multi_hop_033 | Which continent includes the country where Stephen Hawking attended Trinity Hall? | factual | hallucinated | 0.000 | 0.479 | RAG | completed |
| 34 | multi_hop_034 | The head of state of United States was born in a country with what capital? | hallucinated | hallucinated | 0.068 | 0.000 | BASELINE | completed |
| 35 | multi_hop_035 | What capital is associated with the country tied to Time Person of the Year received by Mahatma Gandhi? | hallucinated | hallucinated | 0.080 | 0.000 | BASELINE | completed |
| 36 | multi_hop_036 | What is the capital of the country where Barack Obama was born? | hallucinated | hallucinated | 0.119 | 0.119 | Tie | completed |
| 37 | multi_hop_037 | Identify the capital of the country containing the founder's birthplace of Intel. | factual | hallucinated | 0.000 | 0.000 | RAG | completed |
| 38 | multi_hop_038 | On which continent is the country where Isaac Newton studied at University of Cambridge? | factual | hallucinated | 0.000 | 0.248 | RAG | completed |
| 39 | multi_hop_039 | What is the capital of the country where the head of state of Spain was born? | factual | hallucinated | 0.000 | 0.000 | RAG | completed |
| 40 | multi_hop_040 | Which capital belongs to the country connected to Willard Gibbs Award received by Marie Curie? | hallucinated | hallucinated | 0.545 | 0.000 | BASELINE | completed |
| 41 | multi_hop_041 | What is the capital of the country where Grace Hopper was born? | factual | hallucinated | 0.000 | 0.333 | RAG | completed |
| 42 | multi_hop_042 | What is the capital of the country where the founder of Nvidia was born? | factual | hallucinated | 0.000 | 0.000 | RAG | completed |
| 43 | multi_hop_043 | Identify the continent of University of Cambridge's country for Niels Bohr. | factual | hallucinated | 0.000 | 0.000 | RAG | completed |
| 44 | multi_hop_044 | For Japan, which capital belongs to its head of state's birth country? | factual | hallucinated | 0.000 | 0.000 | RAG | completed |
| 45 | multi_hop_045 | What is the capital of the country associated with the award Angela Merkel received (Grand Cross of the Order of Prince Henry)? | factual | factual | 0.000 | 0.000 | Tie | completed |
| 46 | multi_hop_046 | Name the capital city of the nation in which Galileo Galilei was born. | hallucinated | factual | 0.390 | 0.000 | BASELINE | completed |
| 47 | multi_hop_047 | What is the capital of the nation where Oracle Corporation's founder was born? | factual | factual | 0.000 | 0.000 | Tie | completed |
| 48 | multi_hop_048 | Identify the continent of King's College's country for Alan Turing. | factual | hallucinated | 0.000 | 0.581 | RAG | completed |
| 49 | multi_hop_049 | For France, which capital belongs to its head of state's birth country? | factual | hallucinated | 0.000 | 0.000 | RAG | completed |
| 50 | multi_hop_050 | For Charles Darwin and the award Pour le Mérite for Sciences and Arts order, what is the country's capital? | hallucinated | hallucinated | 0.000 | 0.275 | RAG | completed |
| 51 | multi_hop_051 | Identify the capital of the country containing Winston Churchill's birth location. | factual | factual | 0.000 | 0.000 | Tie | completed |
| 52 | multi_hop_052 | What is the capital of the country where the founder of Tesla, Inc. was born? | factual | hallucinated | 0.000 | 0.982 | RAG | completed |
| 53 | multi_hop_053 | Name the continent of the country that contains University of Paris, where Marie Curie was educated. | factual | factual | 0.000 | 0.000 | Tie | completed |
| 54 | multi_hop_054 | What capital corresponds to the birth-country of the current head of state of Netherlands? | hallucinated | factual | 0.083 | 0.000 | BASELINE | completed |
| 55 | multi_hop_055 | For Albert Einstein and the award Barnard Medal for Meritorious Service to Science, what is the country's capital? | factual | hallucinated | 0.000 | 0.000 | RAG | completed |
| 56 | multi_hop_056 | The birthplace of Alan Turing is in which country's capital city? | factual | hallucinated | 0.000 | 0.966 | RAG | completed |
| 57 | multi_hop_057 | Name the capital city of the founder's birth country for Meta. | factual | hallucinated | 0.000 | 0.000 | RAG | completed |
| 58 | multi_hop_058 | Stephen Hawking studied at St Albans School in a country on which continent? | factual | hallucinated | 0.000 | 0.120 | RAG | completed |
| 59 | multi_hop_059 | For Canada, which capital belongs to its head of state's birth country? | factual | hallucinated | 0.000 | 0.000 | RAG | completed |
| 60 | multi_hop_060 | What is the capital of the country associated with the award Albert Einstein received (Nobel Prize in Physics)? | factual | factual | 0.000 | 0.000 | Tie | completed |
| 61 | multi_hop_061 | What capital corresponds to the country where Mahatma Gandhi was born? | factual | hallucinated | 0.000 | 0.345 | RAG | completed |
| 62 | multi_hop_062 | What is the capital of the nation where Sony Group's founder was born? | factual | factual | 0.000 | 0.000 | Tie | completed |
| 63 | multi_hop_063 | Which continent includes the country where Marie Curie attended Flying University? | hallucinated | factual | 0.000 | 0.000 | BASELINE | completed |
| 64 | multi_hop_064 | For India, which capital belongs to its head of state's birth country? | factual | hallucinated | 0.000 | 0.000 | RAG | completed |
| 65 | multi_hop_065 | What capital is associated with the country tied to Fellow of the Royal Society received by Winston Churchill? | hallucinated | hallucinated | 0.000 | 0.000 | Tie | completed |
| 66 | multi_hop_066 | Marie Curie was born in a country whose capital is what? | hallucinated | hallucinated | 0.607 | 0.593 | BASELINE | completed |
| 67 | multi_hop_067 | What is the capital of the nation where Cisco's founder was born? | factual | hallucinated | 0.000 | 0.000 | RAG | completed |
| 68 | multi_hop_068 | Name the continent of the country that contains Newnham College, where Rosalind Franklin was educated. | factual | factual | 0.000 | 0.000 | Tie | completed |
| 69 | multi_hop_069 | For Italy, which capital belongs to its head of state's birth country? | hallucinated | hallucinated | 1.000 | 0.000 | BASELINE | completed |
| 70 | multi_hop_070 | For Martin Luther King Jr. and the award Jawaharlal Nehru Award for International Understanding, what is the country's capital? | hallucinated | hallucinated | 0.876 | 0.000 | BASELINE | completed |
| 71 | multi_hop_071 | Identify the capital of the country containing Martin Luther King Jr.'s birth location. | factual | hallucinated | 0.000 | 0.079 | RAG | completed |
| 72 | multi_hop_072 | What is the capital of the country where the founder of Intel was born? | hallucinated | hallucinated | 1.000 | 0.971 | BASELINE | completed |
| 73 | multi_hop_073 | Richard Feynman studied at Princeton University in a country on which continent? | factual | factual | 0.000 | 0.000 | Tie | completed |
| 74 | multi_hop_074 | What is the capital of the country where the head of state of South Africa was born? | hallucinated | hallucinated | 0.193 | 0.000 | BASELINE | completed |
| 75 | multi_hop_075 | What capital is associated with the country tied to Gold Medal of the Royal Astronomical Society received by Albert Einstein? | factual | hallucinated | 0.000 | 0.000 | RAG | completed |
| 76 | multi_hop_076 | What capital corresponds to the country where Isaac Newton was born? | factual | factual | 0.000 | 0.000 | Tie | completed |
| 77 | multi_hop_077 | The founder of Google was born in a country whose capital is what? | factual | factual | 0.000 | 0.000 | Tie | completed |
| 78 | multi_hop_078 | The country of Stephen Hawking's alma mater, University College, Oxford, lies on what continent? | factual | hallucinated | 0.000 | 0.545 | RAG | completed |
| 79 | multi_hop_079 | Which capital is tied to the country of birth of Germany's head of state? | factual | hallucinated | 0.000 | 0.000 | RAG | completed |
| 80 | multi_hop_080 | Identify the capital of the nation associated with Nobel Prize in Chemistry that Marie Curie received. | hallucinated | factual | 0.047 | 0.000 | BASELINE | completed |

## Skipped/Error Diagnostics

- None