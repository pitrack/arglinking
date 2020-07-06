
File: Articles-with-extracted-info.tsv

This is the main file you probably care about. It contains articles with extracted structured information. The columns (tab-delimited) are as follows:

- Full text: The full text of the article
- Json: Extracted information (described below)
- Article title: Self-explanatory
- Article url: Self-explanatory

The JSON field contains the following data (note not all information is available for every article, it depends on what is reported in the text)

Unless otherwise noted, the value of each of the described keys is a dictionary containing "value" (the plain text string), "startIndex" (the character-level offset where the value begins in the full article text), and "endIndex" (the character-level offset where the value ends in the full article text).

-- date-and-time : information about date and time with following fields.
---- city: Self-explanatory
---- state: Self-explanatory, filled in using a drop-down menu, so does not have the start/end/value dictionary format described above
---- details: Additional location information about the shooting (e.g. "at a nightclub" or "parking lot of a school")
---- date: Self-explanatory, filled in using a date-selection tool, so does not have the start/end/value dictionary format described above
---- clock-time: Time of shooting
---- time-day: Coarse-grained time of shooting (e.g. "morning", "late evening")

-- victim-section : information about victim(s). The value of victim-section is a list of dictionaries, each dictionary having the following fields:
----name: Self-explanatory
----age: Self-explanatory
----gender: Self-explanatory, filled in using a drop-down menu, so does not have the start/end/value dictionary format described above
----race: Self-explanatory
----victim-was: list containing up to three values ("injured", "hospitalized", "killed") depending on what the outcome to the victim was.

-- shooter-section : information about shooter(s). The value of victim-section is a list of dictionaries, each dictionary having the following fields:
----name: Self-explanatory
----age: Self-explanatory
----gender: Self-explanatory, filled in using a drop-down menu, so does not have the start/end/value dictionary format described above
----race: Self-explanatory

-- circumstances : additional information about shooting
----type-of-gun: Self-explanatory
----number-of-shots-fired: Self-explanatory

-- radio1-3 : answers to true-or-false questions. Each key is the question, and each value is either "True", "False" or "Not mentioned".

File: Articles-with-extracted-info.tsv

This file contains articles potentially involving gun violence that were scraped using our crawler, and then labeled by humans to indicate whether or not they actually describe gun violence. The fields are as follows:

- Full text: The full text of the article
- Article title: Self-explanatory
- Article url: Self-explanatory
- HIT Type: Interface we used for labeling, either "HeadlineHIT" or "ArticleIDHIT"
- Tag: Whether the article is about gun violence. If HIT Type is "HeadlineHIT" the value is either "yes" or "no". If the HIT Type is "ArticleIDHIT", the value is more fine-grained (descriptions should be self-explanatory, let me know if it is unclear)

File: Events.tsv

File which attempts to combined multiple articles about the same event into a single row. This file is still in "beta" and is not necessarily reliable. If you want to use it, let me know and we can figure out if it will be useful for you or not.
