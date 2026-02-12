# Hallucination Comparison Report

**Generated:** 2025-12-12 23:12:58

## Overview

This report compares two approaches to reducing LLM hallucinations:

1. **RAG (Wikidata)**: Retrieves facts from Wikidata before responding
2. **Prompt-Only**: Uses an anti-hallucination system prompt without retrieval

## Results Table

### Vectara Hallucination Model Results

| # | Question | RAG Score | RAG | Prompt Score | Prompt | Winner |
|---|----------|-----------|-----|--------------|--------|--------|
| 1 | Who is Albert Einstein? | 0.937 | ✅ | 0.280 | ❌ | RAG |
| 2 | When was Marie Curie born and what were ... | 0.752 | ✅ | 0.105 | ❌ | RAG |
| 3 | What is the capital of France? | 0.829 | ✅ | 0.763 | ✅ | RAG |
| 4 | What is the relationship between Alan Tu... | 0.985 | ✅ | 0.958 | ✅ | RAG |
| 5 | Tell me about the collaboration between ... | 0.962 | ✅ | 0.955 | ✅ | RAG |
| 6 | Describe the joint research between Eins... | 0.628 | ✅ | 0.614 | ✅ | RAG |
| 7 | Compare the contributions of Ada Lovelac... | 0.617 | ✅ | 0.492 | ❌ | RAG |
| 8 | What organization did Alan Turing work f... | 0.964 | ✅ | 0.981 | ✅ | Prompt-Only |

### RAGTruth Span-Level Detection Results

| # | Question | RAG Score | RAG Spans | RAG | Prompt Score | Prompt Spans | Prompt | Winner |
|---|----------|-----------|-----------|-----|--------------|--------------|--------|--------|
| 1 | Who is Albert Einstein? | 0.000 | 0 | ✅ | 0.154 | 3 | ❌ | RAG |
| 2 | When was Marie Curie born and what ... | 0.000 | 0 | ✅ | 0.196 | 3 | ❌ | RAG |
| 3 | What is the capital of France? | 0.000 | 0 | ✅ | 0.000 | 0 | ✅ | Tie |
| 4 | What is the relationship between Al... | 0.000 | 0 | ✅ | 0.000 | 0 | ✅ | Tie |
| 5 | Tell me about the collaboration bet... | 0.000 | 0 | ✅ | 0.000 | 0 | ✅ | Tie |
| 6 | Describe the joint research between... | 0.000 | 0 | ✅ | 0.059 | 1 | ❌ | RAG |
| 7 | Compare the contributions of Ada Lo... | 0.000 | 0 | ✅ | 0.390 | 4 | ❌ | RAG |
| 8 | What organization did Alan Turing w... | 0.484 | 1 | ❌ | 0.000 | 0 | ✅ | Prompt-Only |

### AIMon HDM-2 Sentence-Level Detection Results

| # | Question | RAG Severity | RAG Sentences | RAG | Prompt Severity | Prompt Sentences | Prompt | Winner |
|---|----------|--------------|---------------|-----|-----------------|------------------|--------|--------|
| 1 | Who is Albert Einstein? | 0.437 | 0 | ✅ | 0.918 | 1 | ❌ | RAG |
| 2 | When was Marie Curie born and ... | 0.543 | 0 | ❌ | 0.359 | 0 | ✅ | Prompt-Only |
| 3 | What is the capital of France? | 0.065 | 0 | ✅ | 0.009 | 0 | ✅ | Prompt-Only |
| 4 | What is the relationship betwe... | 0.065 | 0 | ✅ | 0.226 | 0 | ✅ | RAG |
| 5 | Tell me about the collaboratio... | 0.711 | 1 | ❌ | 0.656 | 1 | ❌ | Prompt-Only |
| 6 | Describe the joint research be... | 0.385 | 1 | ✅ | 0.945 | 1 | ❌ | RAG |
| 7 | Compare the contributions of A... | 0.377 | 1 | ✅ | 0.727 | 1 | ❌ | RAG |
| 8 | What organization did Alan Tur... | 0.922 | 1 | ❌ | 0.496 | 0 | ✅ | Prompt-Only |



## Summary Statistics (Vectara Model)

| Metric | RAG (Wikidata) | Prompt-Only |
|--------|----------------|-------------|
| Total Tests | 8 | 8 |
| Factual Responses | 8 | 5 |
| Hallucinations | 0 | 3 |
| Hallucination Rate | 0.0% | 37.5% |
| Average Score | 0.834 | 0.643 |

## Head-to-Head (Vectara)

| Winner | Count |
|--------|-------|
| RAG (Wikidata) | 7 |
| Prompt-Only | 1 |
| Tie | 0 |

## RAGTruth Statistics (Span-Level Detection)

| Metric | RAG (Wikidata) | Prompt-Only |
|--------|----------------|-------------|
| Hallucinations Detected | 1 | 4 |
| Hallucination Rate | 12.5% | 50.0% |
| Avg Hallucination Score | 0.060 | 0.100 |
| Avg Hallucinated Spans | 0.1 | 1.4 |

## Head-to-Head (RAGTruth)

| Winner | Count |
|--------|-------|
| RAG (Wikidata) | 4 |
| Prompt-Only | 1 |
| Tie | 3 |

## AIMon HDM-2 Statistics (Sentence-Level Detection)

| Metric | RAG (Wikidata) | Prompt-Only |
|--------|----------------|-------------|
| Hallucinations Detected | 3 | 4 |
| Hallucination Rate | 37.5% | 50.0% |
| Avg Hallucination Severity | 0.438 | 0.542 |
| Avg Hallucinated Sentences | 0.5 | 0.5 |

## Head-to-Head (AIMon)

| Winner | Count |
|--------|-------|
| RAG (Wikidata) | 4 |
| Prompt-Only | 4 |
| Tie | 0 |


## Detailed Results


### Test 1: Basic biographical question about a well-known scientist

**Question:** Who is Albert Einstein?

**Ground Truth:**
```
Albert Einstein was a German-born theoretical physicist who developed the theories of special relativity and general relativity, provided the quantum explanation of the photoelectric effect, and contributed foundational work to statistical mechanics, Brownian motion, and early quantum theory. He received the 1921 Nobel Prize in Physics for his explanation of the photoelectric effect. He was born on 14 March 1879 in Ulm, Kingdom of Württemberg (German Empire), and died on 18 April 1955 in Princeton, New Jersey, United States. He was a human, male. His parents were Hermann Einstein and Pauline Koch. He was married to Mileva Marić (1903–1919) and later to Elsa Einstein (1919–1936). His children were Hans Albert Einstein, Eduard Einstein, and his daughter Lieserl Einstein, whose early life is documented through correspondence. He held German citizenship at birth, became Swiss in 1901, and American in 1940. He studied at ETH Zurich, earning a teaching diploma in physics and mathematics, and completed his doctorate at the University of Zurich under Alfred Kleiner. He worked at the Swiss Patent Office in Bern and held academic positions at the University of Zurich, Charles University in Prague, ETH Zurich, the Kaiser Wilhelm Institute for Physics in Berlin, and later the Institute for Advanced Study in Princeton. His notable scientific contributions include special relativity, general relativity, mass–energy equivalence (E=mc²), the Einstein field equations, the explanation of the photoelectric effect, theoretical work on Brownian motion, Einstein coefficients, Bose–Einstein statistics (with S. N. Bose), and the Einstein–Rosen bridge (with Nathan Rosen). His doctoral students included Nathan Rosen and Ernst G. Straus, and he collaborated with scientists such as Marcel Grossmann, S. N. Bose, and Leo Szilard. He received major recognitions including the Copley Medal in 1925 and the Max Planck Medal in 1929. In 1905, his Annus Mirabilis, he published four revolutionary papers that transformed modern physics. He advocated for civil rights in the United States, supported W. E. B. Du Bois, signed the Einstein–Szilard letter in 1939 urging U.S. research into nuclear chain reactions, supported the establishment of the Hebrew University of Jerusalem, and later declined the presidency of Israel in 1952. His brain was removed for scientific study after his death, an action that is historically documented and widely discussed. The chemical element einsteinium (atomic number 99) was named in his honor.
```

#### RAG Model (✅ FACTUAL, Score: 0.937)

**Response:**
Albert Einstein (1879 – 1955) was a German‑born theoretical physicist. He was born on 14 March 1879 in Ulm and died on 18 April 1955. His occupations included physicist, and he held citizenship in Germany, the Kingdom of Württemberg, the Weimar Republic, Switzerland, and the United States. Notable works include quantum mechanics, general relativity, special relativity, mass–energy equivalence, and the theory of relativity.

#### Prompt-Only Model (❌ HALLUCINATION, Score: 0.280)

**Response:**
Albert Einstein (1879 – 1955) was a German‑born theoretical physicist who is best known for developing the theory of relativity, including the famous equation E = mc². His work transformed the understanding of space, time, and gravity, and he made foundational contributions to quantum theory, statistical mechanics, and cosmology. In 1921 he received the Nobel Prize in Physics for his explanation of the photoelectric effect. Einstein emigrated to the United States in 1933, where he taught at the Institute for Advanced Study in Princeton, New Jersey, and became an outspoken advocate for civil rights, nuclear disarmament, and international cooperation.

#### RAGTruth Evaluation

| Model | Status | Score | Spans |
|-------|--------|-------|-------|
| RAG | FACTUAL | 0.000 | 0 |
| Prompt-Only | HALLUCINATED | 0.154 | 3 |

**Prompt-Only Hallucinated Spans:**
- "and cosmology"
  - Reason: The source does not mention Einstein making foundational contributions to cosmology; this claim cannot be verified from the provided context.
- "emigrated to the United States in 1933"
  - Reason: The source states Einstein worked in the United States and became a citizen in 1940, but does not provide the 1933 emigration date.
- "nuclear disarmament, and international cooperation"
  - Reason: The source notes his civil‑rights advocacy and the Einstein–Szilard letter, but does not mention advocacy for nuclear disarmament or international cooperation.

**Analysis:**
- RAG: All statements in the answer are directly supported by the ground truth and fetched entity data; no unsupported or contradictory information is present.
- Prompt-Only: The response is mostly accurate but includes three statements not supported by the source: a claim about contributions to cosmology, a specific emigration year, and advocacy for nuclear disarmament and international cooperation. These are hallucinations under the strict grounding criteria.

#### AIMon HDM-2 Evaluation

| Model | Status | Severity | Sentences |
|-------|--------|----------|-----------|
| RAG | FACTUAL | 0.437 | 0 |
| Prompt-Only | HALLUCINATED | 0.918 | 1 |

**Prompt-Only Hallucinated Sentences:**
- "Einstein emigrated to the United States in 1933, where he taught at the Institute for Advanced Study in Princeton, New Jersey, and became an outspoken advocate for civil rights, nuclear disarmament, and international cooperation." (prob: 0.988)

---

### Test 2: Specific biographical facts with dates

**Question:** When was Marie Curie born and what were her major achievements?

**Ground Truth:**
```
Marie Curie was born on 7 November 1867 in Warsaw, then part of the Russian Empire (Congress Poland). She was a pioneering physicist and chemist best known for discovering the elements polonium and radium, developing the theory of radioactivity, establishing experimental techniques for isolating radioactive isotopes, and laying the foundations of nuclear physics, radiochemistry, and medical radiation therapy. She was a human, female. She died on 4 July 1934 in Passy, France, from aplastic anemia caused by prolonged exposure to radiation. Her parents were Władysław Skłodowski and Bronisława Skłodowska. She married Pierre Curie in 1895 and remained married until his death in 1906. She had two daughters, Irène Joliot-Curie and Ève Curie. She studied at the clandestine Flying University in Warsaw and later at the University of Paris (Sorbonne), earning degrees in physics and mathematics. She held academic positions at the University of Paris and later directed research at the Radium Institute in Paris, becoming the first woman professor at the Sorbonne. Her scientific fields included radioactivity, experimental physics, and radiation chemistry. She received the 1903 Nobel Prize in Physics (shared with Pierre Curie and Henri Becquerel) for research on radiation phenomena, and the 1911 Nobel Prize in Chemistry for the discovery of radium and polonium and the isolation of radium. She was the first woman to win a Nobel Prize, the first person to win two Nobel Prizes, and the only person to win Nobel Prizes in two different scientific fields (Physics and Chemistry). She directed the development and deployment of mobile X-ray units during World War I, personally training operators and assisting with battlefield radiology. She also played a central role in establishing two major research centers—the Radium Institute in Paris and a parallel institute in Warsaw—both of which became leading hubs for nuclear science and medical radiology.
```

#### RAG Model (✅ FACTUAL, Score: 0.752)

**Response:**
Marie Curie was born on 7 November 1867. Her major achievements include receiving the Nobel Prize in Physics, the Nobel Prize in Chemistry, the Order of the White Eagle, the Davy Medal, the Willard Gibbs Award, and authoring the notable work *Treatise on Radioactivity*.

#### Prompt-Only Model (❌ HALLUCINATION, Score: 0.105)

**Response:**
- **Date of birth:** 7 November 1867 (Warsaw, then part of the Russian Empire, now Poland).  

- **Major, verified achievements:**  
  1. **Nobel Prize in Physics (1903)** – Shared with Pierre Curie and Henri Becquerel for their combined research on the phenomenon of radioactivity.  
  2. **Nobel Prize in Chemistry (1911)** – Awarded for the discovery of the elements **polonium** (named for Poland) and **radium**, and for the isolation of radium in a pure metallic form.  
  3. **Pioneering work on radioactivity** – Developed techniques for measuring radioactive emissions, coined the term “radioactivity,” and demonstrated that radioactivity is a property of atoms.  
  4. **First woman to win a Nobel Prize** and the **first person to win two Nobel Prizes in different scientific fields**.  
  5. **First female professor at the University of Paris (Sorbonne)**, where she established the first laboratory dedicated to the study of radioactivity.  
  6. **Medical applications** – Contributed to the early use of radium in cancer treatment and helped develop mobile X‑ray units used during World War I.  

These points reflect the historically documented contributions of Marie Curie.

#### RAGTruth Evaluation

| Model | Status | Score | Spans |
|-------|--------|-------|-------|
| RAG | FACTUAL | 0.000 | 0 |
| Prompt-Only | HALLUCINATED | 0.196 | 3 |

**Prompt-Only Hallucinated Spans:**
- "coined the term “radioactivity,” and demonstrated that radioactivity is a property of atoms."
  - Reason: The source does not mention that Marie Curie coined the term 'radioactivity' or that she demonstrated radioactivity as a property of atoms; these claims cannot be verified from the provided context.
- "where she established the first laboratory dedicated to the study of radioactivity."
  - Reason: The source states she was the first woman professor at the Sorbonne but does not say she established the first laboratory dedicated to radioactivity; this detail is not supported by the source.
- "Contributed to the early use of radium in cancer treatment"
  - Reason: The source mentions her work on mobile X‑ray units during WWI but does not mention any contribution to early radium cancer treatment; this claim is not verifiable from the provided context.

**Analysis:**
- RAG: All claims in the response are verified by the ground truth and fetched entity data; no hallucinations detected.
- Prompt-Only: The answer is largely accurate regarding birth date and major Nobel‑related achievements, but it adds three unsupported details about coining the term 'radioactivity', founding a dedicated laboratory, and early radium cancer therapy, which constitute hallucinations.

#### AIMon HDM-2 Evaluation

| Model | Status | Severity | Sentences |
|-------|--------|----------|-----------|
| RAG | HALLUCINATED | 0.543 | 0 |
| Prompt-Only | FACTUAL | 0.359 | 0 |

---

### Test 3: Simple geographic fact

**Question:** What is the capital of France?

**Ground Truth:**
```
Paris is the capital and largest city of France.
```

#### RAG Model (✅ FACTUAL, Score: 0.829)

**Response:**
Paris.

#### Prompt-Only Model (✅ FACTUAL, Score: 0.763)

**Response:**
Paris.

#### RAGTruth Evaluation

| Model | Status | Score | Spans |
|-------|--------|-------|-------|
| RAG | FACTUAL | 0.000 | 0 |
| Prompt-Only | FACTUAL | 0.000 | 0 |

**Analysis:**
- RAG: The answer 'Paris' is directly verified by the source context, which states that Paris is the capital of France. No hallucinated or unsupported claims are present.
- Prompt-Only: The response correctly states that Paris is the capital of France, which is fully supported by the source context. No unsupported or invented claims are present.

#### AIMon HDM-2 Evaluation

| Model | Status | Severity | Sentences |
|-------|--------|----------|-----------|
| RAG | FACTUAL | 0.065 | 0 |
| Prompt-Only | FACTUAL | 0.009 | 0 |

---

### Test 4: Mix of real and fictional entity - tests hallucination resistance

**Question:** What is the relationship between Alan Turing and Dr. Helena Vargass?

**Ground Truth:**
```
There is no relationship between Alan Turing and Dr. Helena Vargass because Alan Turing is a real historical figure while Dr. Helena Vargass is not a real person. No historical, academic, scientific, or biographical records contain this name, so no connection of any kind exists. Alan Turing was a human mathematician, logician, cryptanalyst, computer scientist, and theoretical biologist. He was born on 23 June 1912 in Maida Vale, London, England, and died on 7 June 1954 in Wilmslow, Cheshire, England. He was male and held British citizenship. His parents were Julius Mathison Turing and Ethel Sara Turing. He never married and had no children. He studied at Sherborne School, then at King's College, Cambridge, where he completed major mathematical work, and later earned his PhD at Princeton University under Alonzo Church. He worked at King's College Cambridge, the Government Code and Cypher School at Bletchley Park during World War II, the National Physical Laboratory, and the University of Manchester. His fields of work included mathematical logic, computability theory, cryptanalysis, early computer architecture, and developmental biology. His notable achievements included the formulation of the universal Turing machine, foundational contributions to computation theory, major contributions to breaking the German Enigma cipher during World War II, and the proposal of the Turing Test for machine intelligence. He was elected a Fellow of the Royal Society in 1951. In 1952, Turing was prosecuted for homosexual acts and forced to undergo chemical castration; he died by suicide in 1954, and was posthumously pardoned by the British government in 2013. Dr. Helena Vargass is fictional, and therefore no factual properties can be provided for this entity.
```

#### RAG Model (✅ FACTUAL, Score: 0.985)

**Response:**
Helena Vargass cannot be verified, therefore a relationship between Alan Turing and Dr. Helena Vargass cannot be confirmed.

#### Prompt-Only Model (✅ FACTUAL, Score: 0.958)

**Response:**
I cannot verify that Dr. Helena Vargass exists, and I have no verified information about any relationship between her and Alan Turing.

#### RAGTruth Evaluation

| Model | Status | Score | Spans |
|-------|--------|-------|-------|
| RAG | FACTUAL | 0.000 | 0 |
| Prompt-Only | FACTUAL | 0.000 | 0 |

**Analysis:**
- RAG: The answer correctly acknowledges that Helena Vargass cannot be verified and therefore no relationship with Alan Turing can be confirmed, which is fully consistent with the provided source context.
- Prompt-Only: The answer correctly acknowledges the lack of verification for Dr. Helena Vargass and the absence of any relationship with Alan Turing, matching the source information. No unsupported or invented claims are present.

#### AIMon HDM-2 Evaluation

| Model | Status | Severity | Sentences |
|-------|--------|----------|-----------|
| RAG | FACTUAL | 0.065 | 0 |
| Prompt-Only | FACTUAL | 0.226 | 0 |

---

### Test 5: Entirely fictional entities - should refuse to provide details

**Question:** Tell me about the collaboration between Dr. Liora Anstrum and Prof. Armin Delacroix.

**Ground Truth:**
```
There is no collaboration between Dr. Liora Anstrum and Prof. Armin Delacroix because both names refer to fictional individuals. No historical, academic, scientific, or publicly documented persons exist with these names, and therefore no factual properties, biographical details, or real-world collaborations can be provided.
```

#### RAG Model (✅ FACTUAL, Score: 0.962)

**Response:**
Both Dr. Liora Anstrum and Prof. Armin Delacroix cannot be verified.

#### Prompt-Only Model (✅ FACTUAL, Score: 0.955)

**Response:**
I cannot verify that Dr. Liora Anstrum or Prof. Armin Delacroix exist, nor any collaboration between them.

#### RAGTruth Evaluation

| Model | Status | Score | Spans |
|-------|--------|-------|-------|
| RAG | FACTUAL | 0.000 | 0 |
| Prompt-Only | FACTUAL | 0.000 | 0 |

**Analysis:**
- RAG: The response correctly reflects the source information by stating that both individuals cannot be verified, with no unsupported or invented claims.
- Prompt-Only: The answer correctly refrains from asserting any facts about the fictional individuals or a collaboration, matching the source that indicates no such real-world information exists.

#### AIMon HDM-2 Evaluation

| Model | Status | Severity | Sentences |
|-------|--------|----------|-----------|
| RAG | HALLUCINATED | 0.711 | 1 |
| Prompt-Only | HALLUCINATED | 0.656 | 1 |

**RAG Hallucinated Sentences:**
- "Both Dr. Liora Anstrum and Prof. Armin Delacroix cannot be verified." (prob: 0.742)

**Prompt-Only Hallucinated Sentences:**
- "I cannot verify that Dr. Liora Anstrum or Prof. Armin Delacroix exist, nor any collaboration between them." (prob: 0.832)

---

### Test 6: Real scientists mixed with fictional - partial hallucination trap

**Question:** Describe the joint research between Einstein, Bohr, and Dr. Selwyn Hartmere on quantum mechanics.

**Ground Truth:**
```
There was no joint research between Albert Einstein, Niels Bohr, and Dr. Selwyn Hartmere because Dr. Selwyn Hartmere is not a real person and therefore no collaboration involving this fictional name ever existed. Albert Einstein and Niels Bohr were real physicists, but they did not conduct joint research together; instead, they engaged in famous scientific and philosophical debates about quantum mechanics, especially at the Solvay Conferences in the 1920s and 1930s. Their interactions were intellectual arguments over foundational issues such as determinism and the Copenhagen interpretation, not co-authored work or shared experiments. Albert Einstein was a human theoretical physicist born on 14 March 1879 in Ulm, Kingdom of Württemberg, German Empire, and he died on 18 April 1955 in Princeton, New Jersey, United States. He held German, Swiss, and American citizenships, was male, and his occupations included theoretical physicist and professor. His parents were Hermann Einstein and Pauline Koch. He married Mileva Marić and later Elsa Einstein and had two sons, Hans Albert Einstein and Eduard Einstein. He studied at ETH Zurich and held academic positions at the University of Zurich, Charles University in Prague, ETH Zurich, the Kaiser Wilhelm Institute for Physics, and the Institute for Advanced Study in Princeton. His scientific fields included relativity, quantum theory, statistical physics, and cosmology, and his major achievements included special relativity, general relativity, the photoelectric effect, and foundational contributions to early quantum physics. He received the 1921 Nobel Prize in Physics, the Copley Medal, and the Max Planck Medal. Niels Bohr was a human theoretical physicist born on 7 October 1885 in Copenhagen, Denmark, where he also died on 18 November 1962. He held Danish citizenship, was male, and his parents were Christian Bohr and Ellen Adler Bohr. He married Margrethe Nørlund and had children including Aage Bohr, who became a Nobel laureate. Niels Bohr studied at the University of Copenhagen and worked with J. J. Thomson and Ernest Rutherford in Cambridge and Manchester. He founded the Institute for Theoretical Physics (now the Niels Bohr Institute) at the University of Copenhagen. His scientific fields included atomic structure, quantum theory, and nuclear physics. His major achievements included the Bohr model of the atom, foundational contributions to quantum mechanics, and the principle of complementarity. He received the 1922 Nobel Prize in Physics and numerous international honors, and he later contributed as a scientific advisor to the Allied nuclear effort during World War II. Because Dr. Selwyn Hartmere does not exist, no factual properties or collaborations can be attributed to this name.
```

#### RAG Model (✅ FACTUAL, Score: 0.628)

**Response:**
Dr. Selwyn Hartmere cannot be verified in Wikidata. No collaborative research involving Albert Einstein (Q937) and Niels Bohr (Q7085) together with a third verified researcher is recorded; the only documented interaction between Einstein and Bohr is the Bohr–Einstein debates, not joint research.

#### Prompt-Only Model (✅ FACTUAL, Score: 0.614)

**Response:**
Einstein and Bohr are historically documented as having engaged in extensive debates over the foundations of quantum mechanics, most famously at the Solvay Conferences in the 1920s and 1930s. These discussions centered on issues such as the completeness of quantum theory, the principle of complementarity, and the Einstein‑Podolsky‑Rosen paradox.

I cannot verify that a researcher named Dr. Selwyn Hartmere existed or that he participated in any joint research project with Einstein and Bohr. Consequently, there is no verified record of a collaborative study involving all three individuals.

#### RAGTruth Evaluation

| Model | Status | Score | Spans |
|-------|--------|-------|-------|
| RAG | FACTUAL | 0.000 | 0 |
| Prompt-Only | HALLUCINATED | 0.059 | 1 |

**Prompt-Only Hallucinated Spans:**
- "the Einstein‑Podolsky‑Rosen paradox"
  - Reason: The source context does not mention the Einstein‑Podolsky‑Rosen paradox; therefore this claim cannot be verified from the provided information.

**Analysis:**
- RAG: All statements in the response are fully supported by the source context; no unsupported or invented claims are present.
- Prompt-Only: All statements about Einstein and Bohr’s debates and the non‑existence of Dr. Selwyn Hartmere are supported by the source. The only unsupported claim is the reference to the Einstein‑Podolsky‑Rosen paradox, which is not present in the source and thus constitutes a hallucination.

#### AIMon HDM-2 Evaluation

| Model | Status | Severity | Sentences |
|-------|--------|----------|-----------|
| RAG | FACTUAL | 0.385 | 1 |
| Prompt-Only | HALLUCINATED | 0.945 | 1 |

**RAG Hallucinated Sentences:**
- "Dr. Selwyn Hartmere cannot be verified in Wikidata." (prob: 0.715)

**Prompt-Only Hallucinated Sentences:**
- "I cannot verify that a researcher named Dr. Selwyn Hartmere existed or that he participated in any joint research project with Einstein and Bohr." (prob: 0.781)

---

### Test 7: Comparison of two related historical figures

**Question:** Compare the contributions of Ada Lovelace and Charles Babbage to computing.

**Ground Truth:**
```
Ada Lovelace and Charles Babbage made foundational but distinct contributions to the early history of computing. Charles Babbage, a human mathematician, inventor, and mechanical engineer, conceived the Difference Engine and designed the Analytical Engine, introducing key architectural principles of modern computers including a memory store, a processing unit (the mill), conditional branching, loops, and programmable operations. (His pioneering work has earned him recognition as the 'father of the computer'.) Ada Lovelace, a human mathematician and writer, studied the Analytical Engine, translated and expanded Luigi Menabrea’s paper on it, and added extensive annotations that included the algorithm for computing Bernoulli numbers—widely regarded as the first published computer program, making Lovelace widely regarded as the world’s first computer programmer. Lovelace also articulated the theoretical insight that the machine could manipulate symbols beyond numerical calculation, envisioning general-purpose computing and early notions of software. Ada Lovelace was born on 10 December 1815 in London, England, died on 27 November 1852 in Marylebone, London, held British citizenship, was female, and her parents were Lord George Gordon Byron and Anne Isabella Milbanke Byron. She married William King-Noel, later Earl of Lovelace, and had three children: Byron, Anne Isabella, and Ralph. She studied mathematics privately under mentors such as Augustus De Morgan and Mary Somerville and worked primarily as an independent thinker. Her fields included mathematics and early computational theory. Charles Babbage was born on 26 December 1791 in London, England, died on 18 October 1871 in London, held British citizenship, was male, and his parents were Benjamin Babbage and Betsy Plumleigh Teape. He married Georgiana Whitmore and had several children. He studied at Trinity College and Peterhouse, Cambridge, later became Lucasian Professor of Mathematics, and worked on mechanical computation, mathematical tables, and precision engineering. He is often regarded as the 'father of the computer' for his pioneering inventions. His designs for the Analytical Engine influenced later generations of computer scientists and historians. Babbage provided the conceptual hardware architecture of computing, while Lovelace contributed the earliest software concepts and theoretical understanding of symbolic computation, and they collaborated intellectually on the Analytical Engine project.
```

#### RAG Model (✅ FACTUAL, Score: 0.617)

**Response:**
Ada Lovelace (1815‑1852) expanded Charles Babbage’s ideas into the first vision of a general‑purpose computer. She translated Luigi Menabrea’s description of Babbage’s Analytical Engine and added extensive notes that included a method for calculating Bernoulli numbers—often called the first published computer program. Lovelace also argued that the Engine could manipulate symbols other than numbers (e.g., music), thereby anticipating the broader capabilities of modern computers.

Charles Babbage (1791‑1871) originated the concept of a digital, programmable computer. He designed the Difference Engine, the first mechanical computer for calculating tables, and the more ambitious Analytical Engine, which incorporated a programmable architecture and, via a principle borrowed from the Jacquard loom, could be instructed to perform arbitrary calculations. He also devised early computer printers and is frequently called the “father of the computer.”

**Comparison**  
- **Invention vs. Programming:** Babbage created the hardware concepts (Difference Engine, Analytical Engine); Lovelace provided the first software‑level work (algorithmic notes, program for Bernoulli numbers).  
- **Scope of Vision:** Babbage’s designs focused on mechanised calculation; Lovelace broadened the vision to include non‑numeric data and general‑purpose computation.  
- **Legacy:** Babbage is credited with inventing the first mechanical computers; Lovelace is credited as the first computer programmer and for articulating the potential of computers beyond pure arithmetic.

#### Prompt-Only Model (❌ HALLUCINATION, Score: 0.492)

**Response:**
**Charles Babbage (1791‑1871)**  
- Designed the **Difference Engine** (1822), a mechanical calculator intended to compute polynomial tables automatically.  
- Conceived the **Analytical Engine** (1837), a general‑purpose mechanical computer with a central “store” (memory), a “mill” (CPU), a punched‑card input system, and the ability to perform conditional branching and loops.  
- Provided the architectural blueprint for modern computers, introducing concepts such as a programmable instruction set and separation of data storage from processing.

**Ada Lovelace (1815‑1852)**  
- Authored the extensive **Notes on the Analytical Engine** (published 1843), which included the first published algorithm intended to be executed by a machine—calculating Bernoulli numbers.  
- Recognized that the Engine could manipulate symbols beyond numbers, foreseeing applications in music, art, and scientific modeling.  
- Introduced the idea of **software** as distinct from hardware, describing how the Engine could be programmed to perform a variety of tasks.

**Comparison**  
- **Primary role:** Babbage was the hardware architect; Lovelace was the first theorist of software and broader computational potential.  
- **Innovation focus:** Babbage created the mechanical structures and logical design of a programmable computer; Lovelace articulated how to program such a machine and envisioned its non‑numeric uses.  
- **Legacy:** Babbage’s designs form the conceptual foundation of computer architecture; Lovelace’s notes are regarded as the earliest expression of computer programming and the notion of computers as general‑purpose information processors.

#### RAGTruth Evaluation

| Model | Status | Score | Spans |
|-------|--------|-------|-------|
| RAG | FACTUAL | 0.000 | 0 |
| Prompt-Only | HALLUCINATED | 0.390 | 4 |

**Prompt-Only Hallucinated Spans:**
- "Designed the Difference Engine (1822), a mechanical calculator intended to compute polynomial tables automatically."
  - Reason: The source mentions Babbage conceived the Difference Engine but does not provide the year 1822; the date is therefore unsupported.
- "Conceived the Analytical Engine (1837), a general‑purpose mechanical computer with a central “store” (memory), a “mill” (CPU), a punched‑card input system, and the ability to perform conditional branching and loops."
  - Reason: The source describes the Analytical Engine but does not give the year 1837; the date is not verifiable from the source.
- "Authored the extensive Notes on the Analytical Engine (published 1843), which included the first published algorithm intended to be executed by a machine—calculating Bernoulli numbers."
  - Reason: The source notes Lovelace’s notes and algorithm but does not state the publication year 1843; the date is unsupported.
- "Recognized that the Engine could manipulate symbols beyond numbers, foreseeing applications in music, art, and scientific modeling."
  - Reason: The source says Lovelace saw symbolic manipulation beyond numbers but does not mention specific foresight about music, art, or scientific modeling; this claim is not grounded in the source.

**Analysis:**
- RAG: All statements in the answer are directly supported by the ground‑truth and Wikipedia excerpts; no unsupported or contradictory claims were found.
- Prompt-Only: The response correctly captures the main contributions of Babbage and Lovelace but adds unsupported dates and specific application examples that are not present in the provided source, resulting in hallucinations.

#### AIMon HDM-2 Evaluation

| Model | Status | Severity | Sentences |
|-------|--------|----------|-----------|
| RAG | FACTUAL | 0.377 | 1 |
| Prompt-Only | HALLUCINATED | 0.727 | 1 |

**RAG Hallucinated Sentences:**
- "He also devised early computer printers and is frequently called the “father of the computer.”

**Comparison**  
- **Invention vs." (prob: 0.945)

**Prompt-Only Hallucinated Sentences:**
- "**Comparison**  
- **Primary role:** Babbage was the hardware architect; Lovelace was the first theorist of software and broader computational potential." (prob: 0.820)

---

### Test 8: Specific historical fact requiring precise answer

**Question:** What organization did Alan Turing work for during World War II?

**Ground Truth:**
```
During World War II, Alan Turing worked for the Government Code and Cypher School (GC&CS) at Bletchley Park, the British government's cryptanalytic center responsible for signals intelligence and codebreaking. The organization was founded in 1919, operated at Bletchley Park during the war, and later became the foundation of GCHQ. Alan Turing was a human mathematician, logician, cryptanalyst, computer scientist, and theoretical biologist born on 23 June 1912 in Maida Vale, London, and he died on 7 June 1954 in Wilmslow, Cheshire. He held British citizenship, was male, and his parents were Julius Mathison Turing and Ethel Sara Turing. He never married and had no children. Turing studied at Sherborne School, King's College Cambridge, and earned his PhD at Princeton University under Alonzo Church. He worked at King's College Cambridge, the Government Code and Cypher School at Bletchley Park, the National Physical Laboratory, and the University of Manchester. His fields included mathematical logic, computability theory, cryptanalysis, early computer architecture, and developmental biology. His major achievements included formulating the universal Turing machine, foundational contributions to computation theory, breaking key components of the Enigma cipher during World War II, and proposing the Turing Test for machine intelligence. He was elected a Fellow of the Royal Society in recognition of his scientific contributions. In 1952, Turing was convicted under British law for homosexual acts and subjected to chemical castration; two years later, in 1954, he died by suicide. In 2013, he received a posthumous royal pardon.
```

#### RAG Model (✅ FACTUAL, Score: 0.964)

**Response:**
Alan Turing worked for the Government Communications Headquarters (GCHQ) during World War II.

#### Prompt-Only Model (✅ FACTUAL, Score: 0.981)

**Response:**
Alan Turing worked for the British Government Code and Cypher School (GC&CS) at Bletchley Park during World War II.

#### RAGTruth Evaluation

| Model | Status | Score | Spans |
|-------|--------|-------|-------|
| RAG | HALLUCINATED | 0.484 | 1 |
| Prompt-Only | FACTUAL | 0.000 | 0 |

**RAG Hallucinated Spans:**
- "Government Communications Headquarters (GCHQ)"
  - Reason: Contradicts source context: Turing worked for the Government Code and Cypher School (GC&CS) during WWII, not GCHQ, which only succeeded GC&CS after the war.

**Analysis:**
- RAG: The response incorrectly identifies Turing's WWII employer as GCHQ, which is not supported by the source and directly conflicts with the stated fact that he worked for GC&CS at Bletchley Park during the war.
- Prompt-Only: All claims in the response are directly verified by the source context; no hallucinations detected.

#### AIMon HDM-2 Evaluation

| Model | Status | Severity | Sentences |
|-------|--------|----------|-----------|
| RAG | HALLUCINATED | 0.922 | 1 |
| Prompt-Only | FACTUAL | 0.496 | 0 |

**RAG Hallucinated Sentences:**
- "Alan Turing worked for the Government Communications Headquarters (GCHQ) during World War II." (prob: 1.000)

---
