# Log of thoughts during the project
**Note: can contain typos, unstructured text (it is not a well written summary) since its literally my thoughts.**

22.9.2026
- first I identify the problem and think about its formulation and make some assumptions
- the data can be any source of text given that it has correct newlines, then I would write a method to corrupt the data to create the dataset.
    - the corruption should represent the actual distribution of inputs.
    - mostly these corrupted texts will have missing lines due to copy-paste, so wrong lines would come from text width of docs or terminals (when I copy texts from terminals, the text breaks are very frustrating)
    - so we can corrupt the data by first removing all the newlines then sample some column widths and put newlines there.
    - now for the correct newlines, lets assume all lines are either:
        1. one line break 
        2. two line breaks 
    - the labels would be between any gaps that is a space or a line (we assum no text segmentation task, eg. "twowords" -> "two words", that would require a seperate different handle apprach)
        1. NEWLINE 
        2. DOUBLE_NEWLINE
        3. SPACE 
        4. JOIN (a word can be split like "wo rd" -> JOIN -> "word")
- for the metric, we can use f1 on acc and recall
- for the data I have chosen wikipedia texts. However, wikimedia/wikipedia seems to contain different languages, so I found a dataset of clean english texts from wikipedia (OVHaiLLM/Clean-Wikipedia-English-Articles), after some manual inspection, I think it is a good dataset to use:
    - it preserve newlines, including 1 and 2 
    - it does not have citations, bib, etc 
    - we should remove '#' since they are not really part of the text for this task (example looks like plain text rather than markdown)
    - \xa0 is non breaking space, we should replace it with a normal space
    - some docs have tables! These are not natural languages so I suppose we should exclude them for this task (there are not many of these)
    - lets do NFKC to clean text 
    - also I will add tests to make sure this cleaning is done correctly
- my lsp is killing me, ill add type hints
