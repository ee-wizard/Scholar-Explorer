# SANTOS: Relationship-based Semantic Table Union Search

Aamod Khatiwada*, Grace Fan*, Roee Shraga, Zixuan Chen,

Wolfgang Gatterbauer, Renée J. Miller, Mirek Riedewald

Northeastern University

{khatiwada.a,fan.gr,r.shraga,chen.zixu,w.gatterbauer,miller,m.riedewald}@northeastern.edu

# ABSTRACT

Existing techniques for unionable table search define unionability using metadata (tables must have the same or similar schemas) or column-based metrics (for example, the values in a table should be drawn from the same domain). In this work, we introduce the use of semantic relationships between pairs of columns in a table to improve the accuracy of union search. Consequently, we introduce a new notion of unionability that considers relationships between columns, together with the semantics of columns, in a principled way. To do so, we present two new methods to discover semantic relationship between pairs of columns. The first uses an existing knowledge base (KB), the second (which we call a “synthesized KB”) uses knowledge from the data lake itself. We adopt an existing Table Union Search benchmark and present new (open) benchmarks that represent small and large real data lakes. We show that our new unionability search algorithm, called SANTOS, outperforms a stateof-the-art union search that uses a wide variety of column-based semantics, including word embeddings and regular expressions. We show empirically that our synthesized KB improves the accuracy of union search by representing relationship semantics that may not be contained in an available KB. This result hints at a promising future of creating a synthesized KBs from data lakes with limited KB coverage and using them for union search.

# ACM Reference Format:

Aamod Khatiwada*, Grace Fan*, Roee Shraga, Zixuan Chen,, Wolfgang Gatterbauer, Renée J. Miller, Mirek Riedewald. 2022. SANTOS: Relationshipbased Semantic Table Union Search. In ,. ACM, New York, NY, USA, 15 pages.

The source code, data, and/or other artifacts have been made available at https://github.com/northeastern-datalab/santos.

# 1 INTRODUCTION

Table search is of growing practical relevance [7, 23, 30, 49]. It allows data scientists to find datasets needed for analysis or training of machine learning algorithms [14, 33]. Google’s dataset search, for example, uses keyword search over metadata which Google incentivizes data owners to use in a standard way [4]. However, when dealing with enterprise data or open data, keyword search via textual queries or over metadata is challenging because metadata may be missing, inconsistent, or incomplete [1, 10, 32, 47]. Hence,

table search for those domains must rely on the data available within given tables instead of relying on well-curated metadata [7, 23].

In this work, we focus on table union search, which is the problem of finding tables that can union with a query table and possibly extend it with more rows. Prior work by Nargesian et al. [33] considers two tables to be unionable if “they share attributes from the same domain". We believe that this is a necessary, but not sufficient condition. In particular, we argue that the relationships modeled by pairs of attributes in a table should share a common semantic meaning. We illustrate this with an example.

Example 1. Consider Tables (a), (b) and (c) in Fig. 1 describing (a) parks, (b) films shown in different parks, and (c) the birth places of famous people, respectively. Suppose the user supplies Query Table (a) and Tables (b), (c) reside in a data lake. The user wants to find data lake tables that union with the Query Table (a). Typically, union search has been defined to permit either the full query table or a projection of it to union with the full data lake table or a projection of it [3]. Now, if we only consider attribute unionability, Table (c) may be considered the most unionable table with three unionable attributes (Person, Birthplace, Country) having high attribute unionability scores with the attributes (Supervisor, City and Country) in the query table. This may be true if we are considering data values [33] or if we consider the attribute names and values [3, 21, 24]. However, if we consider the relationships in the tables, then we would notice that in Query Table (a), City is where the Park is located (and the person in Supervisor works), but in Table (c), Birthplace contains cities where the people in the Person column were born. Deciding unionability based only on attribute unionability, without considering the semantics of the relationships between attributes, can lead to false positives. Even worse, it can lead to the omission of good answers like Table (b) which has information about parks and the city in which they are located, but might be considered less unionable than (c) based solely on column semantics.

We present a new definition of unionability based on both the attribute semantics and the relationship semantics between attributes. Relationships have been used to a limited extent in web table search. Their tables are often assumed to be “entity tables” (i.e., all attributes are properties of a known common entity represented by a single attribute) [7, 8, 28] or they use a subject column and a single relationship from that column to search for tables [43]. However, we do not assume that every table has a single subject attribute that all other attributes describe, nor that we can automatically detect such an attribute. While the entity-table assumption may be mostly true in web tables, data lake tables (in both open data and enterprise lakes) tend to be much wider and may describe the interplay between more than one entity.

![](images/f31fb0afbf561eeecdc9e8e3d7b9beaee0abbb820f49f18903b17f606da0c30f.jpg)

![](images/a993fed8174ebe61c20cea05a1d9fb9bf2b42d5b77d286587ddb010c9077c3cc.jpg)  
(b)

![](images/85e70eb421bde90d342c4a934a974ec6e65958e1b19b7668b240ade71a90d7d5.jpg)  
(c)

![](images/82d191f95c115e1b87ff12f14ee9daef43387fa7732c0a7fe0ae9ac6524d8aae.jpg)

![](images/049941e1d036257b3eec88594131d0cf490b5f7a6f59a773ee7459e7d59b3379.jpg)  
(e)

![](images/756bf6e29c3ac4c9415df0829599b6b335fcca306ef73e8539804b3d08ce0800.jpg)  
  
Figure 1: (a) A table about parks. Relationships between Park Name and Supervisor (ledBy) and between Park Name and City (locatedIn) are found and depicted by the solid lines above the table. We have also found a relationship between Country and City (locatedIn). (b) A table about parks and films shown in the park. Some of the relationships we found are shown by solid and dashed lines above the table. (c) A table about famous people. (d) A semantic graph for Table (a) with the relationships found related to the root Park Name. (e), (f) Semantic graphs for Tables (b), (c), respectively.

In this work, we provide a new definition of table union search. Given a query table $Q$ and a set of data lake tables $\mathcal { T }$ , the top- $k$ table union search problem is to find the best $k$ tables from $\mathcal { T }$ that can be unioned with the query table by taking the column and relationship semantics of all involved tables into account.

Example 2. The columns City and Country in Table (a) are unionable with Birthplace and Country in Table (c) because the columns are unionable and their relationships are the same ( locatedIn). If a user is interested in amassing a collection of cities and their properties like location, then these tables are unionable. However, if a user wants to collect a set of park supervisors and their properties (the city and country in which they work), then Tables (a) and (c) are not unionable. This is because the relationship between Supervisor and City (worksIn) is not the same as the relationship between Person and Birthplace (bornIn). A user may consider Table (b) to be unionable with Table (a), but only on ??(Park Name, City, Country). If we only consider attribute semantics, we might mistakenly say Park Name, Supervisor, City is unionable with Park Name, Film Director, Park City. But the relationship semantics reveal that Supervisor has a different relationship to Park Name than Film Director has to Park Name.

# 1.1 Contributions

Our contributions can be summarized as follows.

• Relationship-based Semantic Table Union Search. We present a new definition of table union search that comprehensively integrates column and relationship semantics.   
• KB Solution. We present a solution to the relationship-based semantic table union search problem that uses a knowledge base (KB) to determine column and relationship semantics. Our solution creates semantic graphs for tables whose nodes are columns and edges are binary relationships found in the KB (depicted in Fig. 1(d)-(f) and explained in Sec. 3). We present a scoring function to match the semantic graph of a data lake table to that of a query table.   
• Synthesized KB Solution. We present a second, novel solution to the search problem that uses the data lake itself to determine if a

set of columns have the same or similar semantics. Our solution determines if the binary relationship between two columns has similar semantics with other binary relationships in the data lake. We show how this synthesized KB can be used to create semantic graphs and solve the table union search problem.

• Empirical Evaluation. Our two solutions were designed to be used in concert with the Synthesized KB compensating for a curated KB that has only partial coverage over a data lake (which is typical when using both open KBs and proprietary enterprise KBs). We show that when used together, our solutions outperform a state-ofthe-art table union baseline. We also experiment with using each solution independently. Finally, we use the Synthesized KB together with different sized samples of the curated KB to understand how accurate our solution is as the coverage of the curated KB decreases.   
• Benchmarks. We develop (and openly share) three new relationship union benchmarks. In the first, we reuse part of the TUS benchmark [33] and label it with relationship semantics. In the second, we use real open data tables and queries, and label a groundtruth. The third is a larger data lake with open data tables and queries, where we only label the query results that either our technique or a baseline technique returns. Hence, this larger benchmark can be used to evaluate precision and efficiency, but not recall.

# 2 RELATED WORK

There is a rich literature on table union search that started with work on web tables. We begin by describing the state-of-the-art in table union search (Sec. 2.1), after which we discuss related work on attribute annotation (Sec. 2.2) and other related work (Sec. 2.3).

# 2.1 Searching for Unionable Tables

Cafarella et al. [5] uses search-style keyword queries and returns a ranked list of relevant web tables. They mainly use tf-idf score and mean string length difference to find the similarity between the columns and combine them to infer the table similarity.

Sarma et al. [7] defines the problem of finding unionable web tables as an entity complement problem. Two tables are deemed unionable if they share similar schemas and a “subject” column,

which is a single column that contains the entities the table is about. This work assumes that each table has a single subject column, a common trait of web tables, but as we argued in Sec. 1 is a limiting assumption in data lakes and open data. Also, this approach relies solely on an existing KB.

Lehmberg et al. [20] uses attribute labels and value overlap between attributes to determine table matching. They build on work from Ling et al. [24] that relies on web tables having identical or similar schemas and uses value overlap. However, attribute labels can be ambiguous or missing in data tables [32], so we cannot rely on such metadata to be complete and consistent.

Nargesian et al. [33] considers two tables to be unionable if they have a bipartite matching between a subset of their columns. This approach generates attribute unionability scores using three different statistical tests, one of which is semantic unionability that makes use of an existing KB. Since they also look at set overlap and word embeddings, they do not rely exclusively on a KB. However, they do not consider the semantics of relationships between columns as discussed in Ex. 1. This in turn, may yield false positive results when searching for unionable tables.

Fernandez et al. [11] introduces SemProp, which links related tables by creating DAGs of table elements and external sources such as ontologies and embeddings, connected by semantic and syntactic links. Although the goals are similar, we solely leverage the data values in tables whereas SemProp mainly uses the tables’ schemas and names to find related tables.

Bogatu et al. [3] proposes an attribute-unionability framework similar to Nargesian et al. [33] that adds attribute name similarity, regular expression similarity, and distribution similarity to determine the relatedness between tables. We focus on attribute values, and also include relationship and type hierarchy semantics.

# 2.2 Annotating Attributes and Relationships

Supervised Approaches. Sherlock is a supervised technique to annotate attributes with 78 semantic types [19]. Zhang et al. extended Sherlock with SATO, a hybrid machine learning model that uses topic modeling and structured learning [46] Recently, Suhara et al. introduced a multi-task learning approach called Doduo [40] that fine-tunes a BERT-base language model [9] and predicts the column types and the relationship types between columns using column values. These approaches, along with other supervised approaches [23, 42] require training over annotated data. Due to the diversity of data lakes (and the large number of possible column and relationship types) and the lack of comprehensive training data, it is difficult to employ a supervised approach to solve the table union problem.

Unsupervised Approaches. There are several unsupervised approaches that annotate columns using KB’s. Mulwad et al. [31] and Syed et al. [41] base their work on KBs to find type labels for column headers. Mazumdar and Zhang propose TableMiner+ [28, 48], which detects a subject column and incrementally finds attribute values in a KB to classify the columns. Venetis et al. [43] extracts data from the web and constructs an isA database for types and a relation database for properties (relationships) to provide a wide coverage of types and relationships. These existing unsupervised approaches rely on information that is sparse in a data

lake, such as column headers. Our approach uses both an existing and a synthesized KB to find column and relation semantics from only attribute values. Also, we make use of a KB type hierarchy to expand the range of union search.

# 2.3 Other Related Work

Embeddings. Nargesian et al. [33] introduced the used of word embeddings for table union search. Cappuzzo et al. [6] proposed to create relational embeddings for each table based on neighboring values in rows and columns. They consider a few data integration tasks, but not unionability and require significant training data.

Domain Discovery. Ota et al. [34] introduce D4, an unsupervised algorithm that uses co-occurrence of values in different tables to discover the set of values that belong to the same domain. Our approach also makes use of co-occurrence of values when creating our synthesized KB, and we not only use co-occurrence of values in a column, but also consider the co-occurrence of values in binary relationships to discover relationship domains.

Semantic Search over Structured Data. Galhotra et al. [14] proposes S3D, a system that finds related tables, rows, and columns from KBs and datasets. S3D’s search method is similar to ours, but they do not account for low KB coverage.

# 3 RELATIONSHIP UNIONABILITY

We now introduce the building blocks of SANTOS (SemANtic Table uniOn Search). SANTOS determines unionability based on not just the semantics of columns (column semantics), but also the semantics of binary relationships between columns (relationship semantics).

Column Semantics. Like previous approaches to semantic annotation or column-type discovery in data lakes [19, 34, 46], we associate each column with a set of semantic annotations. An annotation may be a type in a KB or, when using unsupervised techniques [33, 34], we may determine that a set of columns have the same, but unknown semantics (for example, they are attribute unionable [33]). We call this column semantics. As an example, in Tables (a) and (b) of Fig. 1, even if no known semantics can be found for Park information, we can still determine that the first columns share the same (but unknown) semantics. As the discovery process is uncertain, each annotation or type in the column semantics is associated with a confidence score.

Definition 3 (Column Semantics). Each column ?? in a table ?? has a column semantics (denoted ???? (??)) which is a set of annotations each defining a conceptual domain to which the values in the column may belong. Each annotation $a \in C S ( c )$ has a confidence score between 0 and 1 that reflects the confidence in the inclusion of the annotation ?? in $C S ( c )$ .

Example 4. In Table (b) of Fig. 1, CS(Park Name) may have a set of types {Place, Tourist Attraction, Park} and CS(Film Title) may have {Creative Work, Movie}. The confidence score will reflect our confidence in each type, for example if, hypothetically, more values in column Park Name map to type Tourist Attraction and only a few to the more specialized type Park, then a method may assign a lower confidence score to Park. There may be other columns for which the column semantics is the empty set.

Relationship Semantics. In addition to column semantics used in previous table union search approaches [3, 33], we use relationship semantics to guide the search. Note that a relationship is any binary relation between column values (in particular, it does not need to be a function). In SANTOS, each pair of columns is associated with a set of relationships, each with a confidence score. Using a KB, we may assign a column pair with a known relationship (for example, a property) in the KB [18, 36], or we may use unsupervised techniques to determine that a set of column pairs have the same (but unknown) relationship.

Definition 5 (Relationship Semantics). Each pair of columns ??1, ??2 in a table ?? has a relationship semantics (denoted $R S ( c _ { 1 } , c _ { 2 } ) )$ which is a set of annotations, each defining a conceptual relationship to which the tuples in $\pi _ { c _ { 1 } , c _ { 2 } } ( T )$ may belong. Each annotation $a \in$ $R S ( c _ { 1 } , c _ { 2 } )$ has a confidence score between 0 and 1 that reflects the confidence in the inclusion of the annotation ?? in $R S ( c _ { 1 } , c _ { 2 } )$ .

Example 6. Relationship semantics in Table (b) of Fig. 1 include DirectedBy between Film Title and Film Director, and HasPhone between Park Name and Park Phone. There may be no relationship discovered between Film Title and Park Name, in which case ???? (Film Title, Park Name) $= \emptyset$ .

Semantic Graph. A relationship can be represented as a (Subject, Predicate, Object) triple or an edge in a graph whose nodes are the columns of a table and edges connect pairs of columns. For now, we assume we have an oracle (which may be a KB or any relationship discovery tool) that provides column and relationship semantics. We can form a semantic graph for each table ?? that contains a node for each column and an edge between two columns if they have non-empty relationship semantics.

Definition 7 (Semantic Graph). Given a set of columns $C =$ $\{ c _ { 1 } , c _ { 2 } \ldots c _ { m } \}$ in a table?? , the semantic graph of?? is a graph $S G ( T ) =$ $( V , E )$ with a distinct vertex ???? for each column $c _ { i }$ , labeled with $C S ( c _ { i } )$ . For each column pair $( c _ { i } , c _ { j } ) , i \neq j$ , with non-empty relationship semantics $R S ( c _ { i } , c _ { j } ) \neq 0$ , there is an undirected edge $e _ { i j }$ between $v _ { i }$ and $v _ { j }$ labeled with $R S ( c _ { i } , c _ { j } )$ .

Example 8. Semantic graphs for data lake Table (b) about parks and films and Table (c) about people are shown in Fig. 1 (e), (f) respectively (with column semantics omitted). As seen in (e), semantic graphs of data lake tables may not be connected, depending on the relationship discovery. Some edges are dashed only to make the connection with Tables (b) and (c) more intuitive.

Unionability Search over Semantic Graphs. In SANTOS, a user (a data scientist) provides a Query Table (denoted by ??) and specifies its intent column (denoted by ?? ), which is a column of most interest to the user that forms relationships of interest with other columns. SANTOS creates a semantic graph for the query table that is restricted to being a tree rooted at the intent column, called the Query Semantic Tree. We find relationships from the intent column to other columns in the query table, then transitively search for relationships from these columns. This process allows a user to direct the search to include their intent column. We search for data lake tables that contain a column with a similar (matching)

semantics to the intent column and with similar relationships. More specifically, we will look for a tree within each semantic graph of data lake tables that matches a subtree of the query tree rooted at the intent column. We do not require that the full query semantic tree be covered by a data lake table. We assume a scoring function that considers the strength of column and relationship matching as well as how much of the query tree is matched. We define a precise scoring function in Sec. 6, after we describe how we compute column and relationship semantics. The scoring function captures how closely the Semantic Graph of a data lake table matches with the Query Semantic Tree.

Example 9. Consider Table (a) about parks in Fig. 1 as a query table. Suppose a user’s intent column is Park Name, which has two relationships (1) Park Name–LedBy–Supervisor and (2) Park Name–LocatedIn–City. City forms a third relationship: City–LocatedIn–Country. The semantic graph is depicted in Fig. 1 (d). For brevity, we omit the CS in the figure and include only one RS for each edge. With the intent column I, we can determine if the semantic graph of Table (b) (Fig. 1(e)) contains a subtree that maps to the query semantic tree rooted at I. If the CS for Park Name and City match in (a) and (b) and RS of locatedIn matches locatedIn, then (b) is a candidate unionable table that unions on ΠPark Name, Park City. Notice however the semantic graph Fig. 1(f) of Fig. 1(c) about people. Although there is a possible matching relationship involving Birthplace and City (with City–LocatedIn–Country), (c) is not a candidate unionable table as there is no good match with the user’s intent column.

Definition 10 (SANTOS Top- $\mathbf { \nabla } _ { K }$ Union Search Solution). Given a set of Data Lake Tables $\mathcal { T }$ , a query table $Q$ with a specified intent column ?? , a semantic graph for ??, $S G ( Q )$ , which forms a tree rooted at ?? . We assume a semantic graph matching or scoring function ?? that for each $T \in { \mathcal { T } }$ returns the highest scoring subtree of $S G ( T )$ that matches a subtree of $S G ( Q )$ rooted at ?? along with its score. The SANTOS union search solution is the top-k data lake tables having the highest score modeled by ??.

We introduce two methods to create semantic graphs. Sec. 4 describes one that uses an existing KB, and Sec. 5 introduces another that is designed to be used when there is no or partial KB coverage over the data lake (as commonly seen with enterprise and open data lakes). We discuss how we leverage these methods to create semantic graphs in Sec. 6 and walk through our implementation in Sec. 7. Finally, we present experiments (Sec. 8) using each method independently and together to provide better accuracy.

# 4 SEMANTIC GRAPH CREATION WITH KB

In this section, we present a method for creating semantic graphs using an existing KB. Recall that metadata (like column headers, table name, etc.) may be missing, inconsistent or incomplete in data lake tables [1, 10, 32, 47]. Therefore, we create the semantic graphs by using the cell values only.

# 4.1 Column Semantics

To create column semantics (CS), we associate a set of KB types (called annotations in Definition 3) to each column. Associating

columns with types from a KB is a well-studied problem [19, 38, 43, 46, 48]. Like previous work on unionability [33] and column type detection [34, 40, 46], we only use values within a column to determine the associated KB types. Like others [33, 43], we use both the KB types and type hierarchy to define CS, since there may be tables that match with a query table at a different granular level. Therefore, instead of annotating each column with a single type, we annotate with a set of types. This provides flexibility in matching a data lake table with query tables of different granularity.

Example 11. Consider the Birthplace column in Table(c) of Fig. 1 that describes where people were born. We might assign a more specific type city to this column because the majority of values are cities. However, we might also assign a broader type Place as this column also contains information on places that are not cities (e.g., Texas and Barnet). Consequently, using any single type for the columns in the data lake tables can impact the effectiveness of union search as it may differ from the detected type in the query table. Therefore, we keep a set of types as the CS and select one to use at query time depending on the query table CS. This allows us to match the same data lake table column with the columns from different query tables–some with just cities and some with places.

CS is assigned based on semantic consistency. For example, if a column is assigned a type place, it can also be assigned another granular type city but not music album, which is semantically inconsistent. We make use of the KB type hierarchy to ensure semantic consistency by only assigning types to a column that are in a ISA relationship in the KB type hierarchy.

SANTOS can be used with any open, enterprise-level, or domainspecific KB. We use YAGO 4 [36] (referred to as YAGO hereafter) in our setup which has a single root. The KB root is a generic type and is uninformative to use as part of CS. As YAGO has a large and rich set of types that are direct descendants of the root, we choose to use all direct descendants of this root as the top level types denoted by $A _ { \mathrm { T o p } }$ . These are types that can form the root of CS and are assumed to be semantically disjoint.1 To define the CS(??), we map each value in ?? to the KB. If the value appears in the KB in a type ??, we add ?? and all its ancestors up to a top level type $\in A _ { \mathrm { T o p } }$ to the column semantics candidate set $C S _ { \mathrm { c a n d i d a t e } }$ (??). Note that a single value may map to multiple types (for example in YAGO, Boston2 maps to City and Music Album) and different values may map to different or identical types. After going through all values, if ????candidate (??) contains multiple top level types, then we keep only the top level type (and all its descendants) in $C S _ { \mathrm { c a n d i d a t e } }$ (??) mapped to by the majority of values, which ensures semantic consistency.

Example 12. Consider Birthplace column in Table (c) of Fig. 1 that has 5 unique values. The value Boston is associated with place and creative work, both of which are in $A _ { \mathrm { T o p } }$ . The KB returns both types for the Birthplace column, as well as descendants of place to which at least one value is mapped: {administrative area, city, state} and the descendant of creative work to which at least one value is mapped: {music album}. Notice the majority of

the values are associated with place. Therefore, we select place as the top level type for the Birthplace column and discard creative work and its descendants. Therefore, CS(Birthplace) is the set of types: {place, administrative area, city, state}.

# 4.2 Column Semantics Confidence

We assign a confidence score to each type by taking the product of the frequency score (fs) and granularity score (gs). This score captures the same intuition as the well known measure TF-IDF [2, 37]. TF-IDF measures the importance of a word in a document considering its occurrence and specificity. Here we want to measure the importance of each type. In our case, TF, which is the occurrence of a term in the document [25], aligns with $f s$ and IDF, which captures the specificity and rareness of the term [39], aligns with gs.

Concretely, the frequency score fs(a) of type $a \in C S ( c )$ in column $c$ is the fraction of unique values in $c$ that are mapped to ?? $\left( \left| c _ { a } \right| \right)$ out of all unique values in ?? mapped to the KB $\left( \lvert c _ { \mathrm { K B } } \rvert \right)$ ):

$$
f s (a) = \frac {\left| c _ {a} \right|}{\left| c _ {\mathrm {K B}} \right|} \tag {1}
$$

Example 13. Consider the CS for the column Birthplace of Fig. 1(c) which includes the annotations city and state. These annotations are siblings in the KB, each with an ISA relationship to a parent place Notice however, that only 1 unique data value (Texas) out of 5 is associated with state while 3 are associated with city (Barnet is mapped to neither city nor state but does map to place). To model this difference, we assign frequency scores $f s ( s \tt t a t e ) = 0 . 2$ and $f s ( \mathsf { c i t y } ) = 0 . 6 .$ As all values are mapped to the top level type place, $f s ( { \mathsf { p l a c e } } ) = 1 . 0 .$

A set of entities of a descendant type is always a subset of entities belonging to its ancestor types in the type hierarchy. So, the higher level types are always mapped to by the same or more entities than children types. This is reflected in the frequency score. From the perspective of information theory, the less probable outcome has greater information [26, 39]. In our case, a type that appears more frequently in the KB is less informative. As mentioned earlier, inspired by the well-known tf–idf measure [2], we penalize the frequent ancestor types by using a frequency-based strategy utilizing KB statistics and assign a granularity score (gs) to each type. The basic idea is to count the frequency of each type ?? by counting the entities that map to type ?? and penalize the frequent types. The granularity score is computed in Eq. (2) which uses a log function.

$$
g s (a) = \frac {1}{\operatorname* {m i n} (1 , \log (a . c o u n t))} \tag {2}
$$

To keep gs consistent with fs, gs also ranges from 0 to 1. Note that for rare types with less than 10 entities, the log function returns a value less than 1. Thus, we use the min function in the denominator.

Example 14. Consider the entity Boston, which belongs to the types city and place. In YAGO, over 6 million entities have type place, and $\sim 4 2 , 0 0 0$ entities have type city. Then, city is a more informative type than place, which is an ancestor type. Hence, the granularity scores are $g s ( \mathsf { p l a c e } ) \approx 0 . 1 4$ and $g s ( \mathsf { c i t y } ) \approx 0 . 2 2$ .

We assume that the computations of granularity score and frequency score are independent of one another as the former is based on KB statistics and the latter is based on the semantics of values in a column. Our objective is to prioritize those types that are more specific (i.e. higher granularity score) and also capture the semantics of the column values better (i.e. high frequency score). Thus, given a column ?? with annotation ??, we define the KB Column Semantics Confidence Score by aggregating fs and gs of ??:

$$
C S _ {\mathrm {C O N F}} (c, a) = \left\{ \begin{array}{l l} f s (a) \cdot g s (a) & \text {i f} c \in \text {d a t a - l a k e} T \\ f s (a) & \text {i f} c \in \text {q u e r y} Q \end{array} \right. \tag {3}
$$

Note that to avoid double penalization, we only penalize the top-level types in the data-lake tables.

Example 15. Birthplace in data lake table Fig. 1(c) has ???? with respective $C S _ { C O N F }$ {place:1.0·0.14, administrative area:1.0· 0.17, city: $0 . 6 0 \cdot 0 . 2 2$ , state: $0 . 2 0 \cdot 0 . 3 5 )$ .

# 4.3 Relationship Semantics

We compute relationship semantics (RS) for every pair of string (non-numeric) columns within the query table and within data lake tables. Note that many pairs of columns (like Park Name and Film Director) may not have a semantic relationship represented in the KB. Intuitively, there is at least an indirect relationship between all columns in a table (in this case, the director of a film shown in the park), but in this section, we are only interested in relationships found in the KB. Suppose columns $c _ { 1 }$ and $c _ { 2 }$ in table ?? have nonempty CS (meaning they are annotated with at least one KB type). To determine RS, we determine if a pair of values in $\pi _ { c _ { 1 } , c _ { 2 } } ( T )$ is associated with entities $( e _ { 1 } , e _ { 2 } )$ that have a KB relationship $\boldsymbol { p }$ . The RS Confidence Score $R S _ { \mathrm { C o n F } } ( c _ { i } , p , c _ { j } )$ for a binary relationship $\boldsymbol { p }$ between columns $c _ { i } , c _ { j } \ ( \pi _ { c _ { i } , c _ { j } } ( T ) )$ in $T$ is:

$$
R S _ {\mathrm {C O N F}} \left(c _ {i}, p, c _ {j}\right) = \frac {\left| \left(c _ {i} , c _ {j}\right) _ {p} \right|}{\left| \left(c _ {i} , c _ {j}\right) _ {\mathrm {K B}} \right|} \tag {4}
$$

such that $| ( c _ { i } , c _ { j } ) _ { p } |$ is the number of unique value-pairs with predicate $\mathcal { P }$ from KB, and $| ( c _ { i } , c _ { j } ) _ { \mathrm { K B } } |$ is the total number of unique value pairs mapped to the KB. Note that only the relationship semantics with the maximum score is included in the semantic graph.4

Example 16. In Fig. 1(c), RS(Person,Birthplace) contains the annotation birthplace with confidence score $R S _ { C o N F } = 1 . 0 $

# 5 SYNTHESIZED KB SEMANTIC GRAPH

KBs may have limited coverage over real data lakes. Hence, using only an existing KB (even a set of KBs) to determine CS and RS can lead to low coverage. Our experimental study indicates that YAGO [36], a well-known and maintained KB, covers only $4 2 \%$ of the string cell values in UK open data and $3 4 \%$ in Canada open data. To solve this problem, we propose a novel data-driven approach using the data lake itself, creating a synthesized KB. Our key insight is that we can replace the role of an existing KB in finding CS and RS with a KB that captures co-occurrence information across data lake tables.

To determine CS, instead of mapping values to an existing KB, we now use a mapping to other columns with the same values. Rather than finding actual semantic types, we leverage type co-occurrence across columns to decide their semantic compatibility. To do so, we annotate all values that co-occur in a column or a meaningful relationship with a “synthesized type.” We then determine CS and RS by also considering other column and column pairs (thus their synthesized types) with overlapping values.

# 5.1 Synthesized Column Semantics

Generally, values within the same column share the same semantic types. For example, in Fig. 1(c), all values in the Person column have type person and values in Birthplace share type place. Considering this property and assuming we do not have homographs [22], we start by assigning to each column in the data lake a unique synthesized column semantics with a confidence score equal to 1. For example, considering Table 1 about parks and movies in Fig. 2, we assign synthesized column semantics A, B to the columns. Similarly, we assign D, E and F, G as column semantics to the columns in Tables 2, 3 respectively.

There can also be different columns that share the same column semantics. For instance, consider Tables 1 and 2 in Fig. 2. Notice that columns A and D are both about parks while columns B and E are about movies. We hypothesize that columns with common semantics have overlapping values that share the same types. We want to determine how likely it is for a column ?? to share semantics with column $c _ { j }$ , and thus inherit $C S ( c _ { j } )$ . Formally, along with its own column semantics, Like in Sec. 4.2, we also assign synthesized type ?? from column $c _ { j }$ to column ?? with a confidence score. We define a synthesized column semantics confidence score for column ?? where $C S _ { \mathrm { C o v F } }$ of ?? is the fraction of unique values in ?? that are also in $c _ { j } \ ( | c \cap c _ { j } | )$ over the total unique values in ?? (|?? |). If $c = c _ { j }$ then $C S _ { \mathrm { C o N F } }$ of $a$ is 1.

$$
C S _ {\mathrm {C O N F}} (c, a \in C S (c _ {j})) = \left\{ \begin{array}{l l} 1 & \text {i f} c = c _ {j} \\ \frac {| c \cap c _ {j} |}{| c |} & \text {o t h e r w i s e} \end{array} \right. \tag {5}
$$

Note that as we do not have type hierarchy information for the synthesized column semantics, we assume all synthesized CS are of the same granularity level and set $g s ( a ) = 1$ for each type.

# 5.2 Synthesized Relationship Semantics

For column pairs in a table that share a "meaningful relationship", we assign a synthesized relationship semantics (RS). For example, consider Table 1 of Fig. 2 about films shown at parks, where columns A and B have a meaningful relationship, annotated as $R S ( A , B )$ . Following the same logic as annotating synthesized CS, we consider two column pairs to have the same RS if their value pairs overlap. For instance, column pairs (A, B) and (D, E) in Table 2 of Fig. 2 have overlapping value pairs (depicted in bold text), indicating that they likely share a meaningful relationship. Therefore, along with its own synthesized RS, we also assign a synthesized $\mathrm { R S } p \in R S ( d _ { i } , d _ { j } )$ of column pair $( d _ { i } , d _ { j } )$ to the column pair $\left( c _ { i } , c _ { j } \right) \left( \pi _ { c _ { i } , c _ { j } } ( T ) \right)$ $( c _ { i } , c _ { j } )$ of table

![](images/2c395ea76c32c0541eb0142e7fd709f9318c18ae97ec9db8bd327df83fa09ef8.jpg)  
Figure 2: Synthesized relationship semantics of data lake tables and their respective semantic graphs. Value-pairs bolded and highlighted in orange appear in Tables 1 and 2, and the value-pair italicized and highlighted in blue appear in Tables 2 and 3.

$T$ with synthesized $R S _ { \mathrm { C o N F } }$ given by:

$$
R S _ {\mathrm {C O N F}} \left(c _ {i}, p, c _ {j}\right) = \left\{ \begin{array}{l l} 1 & \text {i f} c _ {i} = d _ {i}, c _ {j} = d _ {j} \\ \frac {\left| \left(c _ {i} , c _ {j}\right) \cap \left(d _ {i} , d _ {j}\right) \right|}{\left| \left(c _ {i} , c _ {j}\right) \right|} & \text {o t h e r w i s e} \end{array} \right. \tag {6}
$$

where $| ( c _ { i } , c _ { j } ) \cap ( d _ { i } , d _ { j } ) |$ is the number of unique overlapping value pairs between distinct column pairs and $| ( c _ { i } , c _ { j } ) |$ is the total unique value-pairs in $( c _ { i } , c _ { j } )$ .

Note that although we may have multiple types in relationship semantics for each column pair (seen in Semantic Graphs in Fig. 2), we use only the type that best matches with the query table during the query phase (Sec. 6). We discuss implementation details for efficient indexing of the synthesized KB in Sec. 7.

# 6 SANTOS UNION SEARCH

In the previous sections, we presented two methods for creating semantic graphs: one relying on a high-quality existing KB (YAGO), the other on our proposed synthesized KB. We now discuss how to compute a unionability score from both semantic graphs.

Scoring function. Recall that SANTOS takes a query table as input from the user and the objective is to generate a ranked list of topk unionable tables. Generating a ranked list of relevant documents for a given query is a well-studied problem in the literature [13, 15]. Recently, Ho et al. formulated a ranking function to rank a list of relevant text documents for a given sentence query [16]. They extract a set of tokens representing the context of the query sentence and each candidate document. Each token is assigned a confidence score. The (normalized) summation of the confidence scores of matching tokens is used to rank the documents. This function is modified and used to rank the relevant web tables for the given quantity queries [17]. We use the same concept to motivate our scoring function without normalization as this does not change the ranking. In our work, the context is represented by matching query table’s column pairs with data lake table’s column pairs and the confidence is represented by their matching quality. Specifically, we match a connected subtree of $Q$ ’s semantic query tree, rooted at the intent column, into the semantic graph of data lake tables. Intuitively, the larger the number of matching column pairs and the higher the confidence that the corresponding semantic types agree, the higher the match score. Note that in the score computation, nodes and edges may each be annotated with multiple possible CS and RS (Sec. 4.1, Sec. 5.2), each with respective confidence scores.

Example 17. Consider table (a) of Fig. 1 about parks. Possible semantics for column pair (Park Name, City) are park-locatedin-place, park-locatedin-city, etc. based on the given KB. When value pairs from these columns are not covered by the given KB, we also obtain semantics from the synthesized KB such as CS(W)-RS(W,X)-CS(X) and CS(Y)-RS(Y,Z)-CS(Z), where CS(W), CS(X), CS(Y) and CS(Z) are the synthesized CS and RS(W,X), RS(Y,Z) the synthesized RS.

With semantics from both the KB and synthesized KB for the same column pair, we propose a 2-step approach for computing the unionability score. First, an “intra-method” technique selects the best semantic match between query and data lake table for each source separately. Then an “inter-method” comparison selects the semantic match that maximizes the overall unionability score.

Let $a _ { 1 } , a _ { 2 } \dots a _ { x } = ( C S ( Q _ { c } ) \cap C S ( T _ { c } ) )$ be the intersecting CS between a query table Q’s column $Q _ { c }$ and a data lake table ?? ’s column $T _ { c }$ given by a semantic graph creation method $G$ (i.e., either KB or the synthesized KB). For each ??, let the corresponding confidence scores be $C S _ { \mathrm { C o n F } } ( Q _ { c } , a _ { i } )$ and $C S _ { \mathrm { C o N F } } ( T _ { c } , a _ { i } )$ , respectively. Since the column semantics assigned to the query table and data lake table are independent of each other, we take a product of confidence scores for each token to get their match score. Then, we select the match that maximizes the score. Intuitively, the match score between $Q _ { c }$ and $T _ { c }$ is determined by the column semantics with the greatest product of confidence scores:

$$
\operatorname {c o l M a t c h} _ {G} \left(Q _ {c}, T _ {c}\right) = \max  _ {i} C S _ {\text {C O N F}} \left(Q _ {c}, a _ {i}\right) \cdot C S _ {\text {C O N F}} \left(T _ {c}, a _ {i}\right). \tag {7}
$$

We determine the relationship match score for column pairs $( Q _ { c 1 } , Q _ { c 2 } )$ in $Q$ and $( T _ { c 1 } , T _ { c 2 } )$ in $T$ analogously as:

$$
\begin{array}{l} r e l M a t c h _ {G} \left(\left(Q _ {c 1}, Q _ {c 2}\right), \left(T _ {c 1}, T _ {c 2}\right)\right) \\ = \max  _ {i} R S _ {\text {C O N F}} \left(Q _ {c 1}, p _ {i}, Q _ {c 2}\right) \cdot R S _ {\text {C O N F}} \left(T _ {c 1}, p _ {i}, T _ {c 2}\right). \tag {8} \\ \end{array}
$$

Here $p _ { 1 } , p _ { 2 } , \ldots , p _ { x } = ( R S ( Q _ { c 1 } , Q _ { c 2 } ) \cap R S ( T _ { c 1 } , T _ { c 2 } ) )$ are the intersecting relationship semantics between the column pairs. Note that depending on the KB, $R S ( T _ { c 1 } , T _ { c 2 } )$ may differ from $R S ( T _ { c 2 } , T _ { c 1 } )$ Therefore, the KB may return $R S ( T _ { c 1 } , T _ { c 2 } )$ for the data lake table and $R S ( Q _ { c 2 } , Q _ { c 1 } )$ for the query table. To model this, we preserve both $R S ( T _ { c 1 } , T _ { c 2 } )$ and $R S ( T _ { c 2 } , T _ { c 1 } )$ for the data lake table. Once we get RS for the query table, we match the data lake table’s RS that maximizes the score with respect to the query table’s RS in Eq. (8).

The overall match score between $Q$ ’s column pair $( Q _ { c 1 } , Q _ { c 2 } )$ and ?? ’s column pair $( T _ { c 1 } , T _ { c 2 } )$ based on method $G$ is then computed as:

$$
\begin{array}{l} p a i r M a t c h _ {G} \left(\left(Q _ {c 1}, Q _ {c 2}\right), \left(T _ {c 1}, T _ {c 2}\right)\right) = c o l M a t c h _ {G} \left(Q _ {c 1}, T _ {c 1}\right) \\ \cdot \operatorname {r e l M a t c h} _ {G} \left(\left(Q _ {c 1}, Q _ {c 2}\right), \left(T _ {c 1}, T _ {c 2}\right)\right) \cdot \operatorname {c o l M a t c h} _ {G} \left(Q _ {c 2}, T _ {c 2}\right) \tag {9} \\ \end{array}
$$

![](images/4f08fcb33b25a16957a9901a1ec3906e321e9637d8afa11a88d382c14ade5f85.jpg)  
Figure 3: Computation of pairMatch score between the semantic tree of the parks table Fig. 1(a) as query table (left) and the parks-and-movies table Fig. 1(b) as data lake table (right). From the data lake table’s semantic graph, we extract the semantic tree rooted at Park Name. The dotted arcs connect the matching columns and relationships.

$$
\begin{array}{l} \text {p a i r M a t t h} _ {\mathrm {K e r}}: \text {p a i r - l o c a t e d I n - c y c l e :} 1. 0 * 0. 7 * 0. 2 5 * 0. 9 5 * 0. 9 4 * 0. 7 8 * 0. 8 8 * 0. 2 2 = 0. 4 8 * 0. 8 9 * 0. 0 6 \\ \text {p a i r M a t t h} _ {\text {S y n t h}}: \text {C S (W) - R S (Y , Z) - C S (X) :} 0. 4 5 * 0. 6 0 * 0. 6 5 * 0. 8 5 * 0. 8 2 * 0. 7 5 = 0. 1 7 * 0. 5 5 \\ \text {p a i r M a t t h : p a r k - l o c a t e d I n - c y c l e :} 0. 4 8 * 0. 8 9 * 0. 0 6 \end{array}
$$

Let ???? and ????????ℎ represent the semantic graph creation methods based on the given and synthesized KB, respectively. Recall that $C S _ { \mathrm { C o N F } }$ based on the given KB is assigned a granularity score for penalizing the top-level types (Eq. (3)). However, due to the absence of a type hierarchy in the synthesized KB, we consider all types to be of same level and assign a granularity score of 1 to each synthesized column semantics (Eq. (5)). To avoid bias, we compare the “inter-method” pair matches ignoring the granularity score.

Formally, let $a _ { 1 }$ and $a _ { 2 }$ be the column semantics selected in ?????????? $\left[ a t c h _ { K B } ( ( Q _ { c 1 } , Q _ { c 2 } ) , ( T _ { c 1 } , T _ { c 2 } ) ) \right.$ . With $f$ denoting a flag such that ?? = 1 iff ???????? ????????ℎ???? ( (????1,????2),(????1,????2)) $\begin{array} { r l r } { f } & { { } = } & { 1 } \end{array}$ $\begin{array} { r l } { \frac { p a i r M a t c h _ { K B } ( ( Q _ { c 1 } , Q _ { c 2 } ) , ( T _ { c 1 } , T _ { c 2 } ) ) } { a \mathfrak { c } ( a , \textup { l } _ { a \mathfrak { c } } / _ { a \mathfrak { c } } ) } } & { { } \geq } \end{array}$ ???? (??1).???? (??2) ???????? ????????ℎ????????ℎ ( (????1, ????2), (????1, ????2)), the match score between $( Q _ { c 1 } , Q _ { c 2 } )$ and $( T _ { c 1 } , T _ { c 2 } )$ is computed as:

$$
\begin{array}{l} p a i r M a t c h \left(\left(Q _ {c 1}, Q _ {c 2}\right), \left(T _ {c 1}, T _ {c 2}\right)\right) = \\ \left\{ \begin{array}{l l} p a i r M a t c h _ {K B} \left(\left(Q _ {c 1}, Q _ {c 2}\right), \left(T _ {c 1}, T _ {c 2}\right)\right) & \text {i f} f = 1 \\ p a i r M a t c h _ {\text {S y n t h}} \left(\left(Q _ {c 1}, Q _ {c 2}\right), \left(T _ {c 1}, T _ {c 2}\right)\right) & \text {o t h e r w i s e} \end{array} \right. \tag {10} \\ \end{array}
$$

Example 18. Let Table (a) in Fig. 1 about parks be the query table $( Q )$ with Park Name as the intent column. Consider Table (b) in Fig. 1 about parks and movies as the data lake table (?? ). There is a possible match between these two tables such that Park Name matches with Park Name and City matches with Park City. We compute the corresponding pair match score as follows. Fig. 3 shows the semantic graphs and the scores involved in selecting the matching pairs. The query table semantic tree is shown on the left and the semantic tree extracted from the data lake semantic graph rooted at Park Name is shown on the right. The semantic graph before extracting the tree is shown in Fig. 1(e). The dotted arcs connect the matching nodes and edges. First, we use Eq. (7) to find the matching column semantics at each node between the tables using existing and synthesized KBs. For

Park Name and Park Name, park and CS(W) are selected from the existing and synthesized KB, respectively. Also for City and Park City, city and CS(X) are selected using the same formula (Eq. (7)). Furthermore, as we have only one relationship semantics for KB, i.e., locatedIn, it is selected as the winner. For the synthesized KB method, RS(Y,Z) wins over RS(W,X) according to Eq. (8). Finally, we use Eq. (10) to perform the inter-method comparison between park–locatedIn–city and CS(W)–RS(Y,Z)–CS(X). As shown in the figure, $a s m a x ( 0 . 4 8 \cdot 0 . 8 9 3 , 0 . 1 6 6 \cdot 0 . 5 5 2 ) = 0 . 4 8 \cdot 0 . 8 9 3$ , we select park–locatedIn–city as the pairMatch between the column pairs.

If a branch is selected from the existing KB, we include the granularity score in the pairMatch score so that when we compare the score between the query table and different data lake tables, the data lake tables matching in the granular types are prioritized over the tables that match just on the top level types. Our pairMatch score may be low if there is no match at the most granular level. One can design a different penalization scheme to penalize the frequent types to change the trend of scoring. However, our objective is to introduce relative difference in the match between the query table and each data lake table rather than the absolute unionability score.

Let there be $m$ matching column pairs between $Q$ ’s semantic tree $S G ( Q )$ rooted at intent column ?? and ?? ’s subtree $S G ( T )$ rooted at ??. We compute the unionability score (??) between ?? and $T$ using

$$
S (Q, T) \quad = \quad \sum_ {i = 1} ^ {m} p a i r M a t c h ((Q. I, Q. c _ {i}), (T. c, T. c _ {i})). \tag {11}
$$

Intuitively, Eq. (11) favors the data lake table that has more matching columns, finer granularity of matching types, and greater probability of matching types.5. The latter two are captured by the individual matching scores while the former is addressed by the summation of such scores. Note that unionability between two tables is often viewed as a binary decision, i.e., either the tables are unionable or not. In a traditional database setting, two tables are unionable iff they have compatible schemata.6 In the data-lake setting, the constraint of aligning a unionable table’s schema exactly with that of a query table is not practical. Existing works [3, 33] have relaxed this in their unionability definitions. However, they solely consider column unionability without considering the relationships and the preservation of the query table’s topics. Thus, by introducing SANTOS, we aim to show the importance of considering relationships when searching for unionable tables that share the same intent as the query table.

# 7 SANTOS IMPLEMENTATION

We now describe SANTOS implementation, namely, the KB use, the synthesized KB creation, and the pre-processing and query phases.

# 7.1 Knowledge Base Implementation

When creating CS and RS using YAGO types, we build four dictionaries for efficient access. First, we store labels and alternate names (synonyms) that describe the corresponding entity, derived

![](images/92f5f7e903d8350817f3612e1e8055542c90fcaa5371a998d3cf8b7c8c10eba3.jpg)

![](images/d0192497c0cf4631803e19421f58c2585d9b7d4c601008df7ab86ec4033c4034.jpg)  
Figure 4: Pipeline of SANTOS. (a) Preprocessing phase: data-lake tables are labeled with types from YAGO or synthesized KB. (b) Query phase: the query table is annotated, and we query the data lake to retrieve and rank unionable tables. Columns highlighted in blue represent the (matching) intent column. The type hierarchy is omitted for simplicity.

from YAGO, mapped to URIs7, in an entity dictionary. 8 We connect tables to the KB by mapping column values to KB labels and alternate names. Note that YAGO permits multiple entities to have the same label or alternate name, so a value may be mapped to entities of different types (e.g. homographs [22]). Second, we build an inheritance dictionary that stores each Top level type and their children types. Next, we use a type dictionary to lookup entities as keys, which are also values in the entity dictionary, with their set of (types, granularity score) as values (e.g. Boston: {(place, 0.14), (city, 0.22),. . . }). Finally, we use a relationship dictionary to store the set of binary relationships (properties) for each value-pair in the KB, similar to the (proprietary) relation database used to recover semantics of webtables [43].

As we index only the necessary KB triples, the total space taken by these dictionaries in main memory is 3.75 GB. We need 965.18 MB to store them persistently. 9 Note that, creating these dictionaries is a one-time task and takes less than 20 minutes in our experimental setup (see Sec. 8.1).

# 7.2 Synthesized KB Implementation

For the synthesized KB, we create a Synthesized Type Dictionary and Synthesized Relationship Dictionary. As we directly use cell values as entities, an entity dictionary is not required. Also, the inheritance dictionary is not needed as we consider all synthesized types to have the same granularity. We will only discuss the creation of the Synthesized Relationship Dictionary for column pairs, since the Synthesized Type Dictionary is created in the same way.

Synthesized Relationship Dictionary. In Sec. 4, we built a relationship dictionary using an existing KB to create annotation to add to the RS. Here, we follow an inverse process, i.e., we first assign a new binary RS to each column pair, then populate a synthesized relationship dictionary with each value-pair in the column pair that

![](images/c496a696af3d66e446a5ac685c160d0a7ad498bc50ac8fd9d9624415da80ce94.jpg)  
Figure 5: Synthesized Relationship Dictionary for Fig. 2.

is not in any existing KB. For each such value-pair, we assign a binary synthesized relationship semantic (RS), which is a distribution of column pairs that exhibit some meaningful relationships among value-pairs. A similar idea exists in topic modeling [35], where some latent topic of a document (value-pair) is represented by a distribution over other documents (column-pairs).

We first assign all value pairs in a column pair with the same synthesized RS. For instance, value-pairs (Brands Park, Moana), (Kells Park, Spider-Man) and (Eckhart Park, Avengers) in Table 1 of Fig. 2 are all assigned $R S ( \mathsf { A } , \mathsf { B } )$ . As we have discussed however (Ex. 6), some binary relationships between columns in a data lake are not meaningful and may be indirect. For example, in Fig. 1 Table (c), value-pairs in Π?? ???????? ????????,???????????????????????? like (Bee Movie, 6748 N. Sacramento Ave.) and (Coco, 5210 W. 64th St.), have a tenuous relationship at best (film shown in a park with this address). Thus, we aim to capture only meaningful binary relationships in which one column functionally determines another. For example, in Fig. 1(b), Park Name functionally determines Park Location, indicating a possible meaningful relationship. We hypothesize that column pairs in a functional dependency are more likely to contain a meaningful semantic relationship that also exists in other tables and may be useful in union search. This also has a benefit of reducing the synthesized relationship dictionary size. We use an existing functional dependency (FD) discovery algorithm called FDEP [12] to find unary FDs (FDs with a single column determinant in binary relationships), and run their bottom-up variant

that first considers all pairwise relationships, then checks if a dependency satisfies an FD. Although FDEP time complexity is quadratic, we only use it offline.

To generate our synthesized relationship dictionary, we first iterate over the data lake tables and store value-pairs in column pairs (not found in the existing KB) that form an FD, and their synthesized RS with a type score of 1, in a lookup dictionary. We then find the type score of each RS to capture the likelihood for the associated value-pair to have that semantics. Type scores, also seen as confidence scores for value-pairs, are also calculated based on the overlap of value pairs in different tables (consistent with Eq. (6)). However, we now consider value-pairs rather than column-pairs, so we calculate the overlap score for each value-pair in a column pair. Thus, we fill the Synthesized Relationship Dictionary (S) with RS from the lookup dictionary and their associated type scores. We clarify the computation using the following example.

Example 19. Consider Tables 1, 2, and 3 in Fig. 2 and S in Fig. 5. First, we assign all value-pairs with ???? (A,B), ???? (D,E) and $R S ( \mathsf { F } , \mathsf { G } )$ with respect to their consisting column pairs with type score 1. Next, consider Table 1 $( R S ( \mathsf { A } , \mathsf { B } ) )$ , which shares two value-pairs (bolded) with Table 2 $( R S ( \mathsf { D } , \mathsf { E } ) ,$ out of three total value-pairs. (Brands Park, Moana) in Table 1 is then also assigned $R S ( \mathsf { D } , \mathsf { E } )$ with a type score of $\textstyle { \frac { 2 } { 3 } }$ . Now consider Table 2, which shares two value-pairs with Table 1 (bolded) and one value-pair with Table 3 (italicized), out of five total value-pairs. Then, (Chopin Park, Trolls) is assigned $R S ( \mathsf { A } , \mathsf { B } )$ with a type score of ${ \frac { 2 } { 5 } } = 0 . 4$ and $R S ( \mathsf { F } , \mathsf { G } )$ with a type score of ${ \frac { 1 } { 5 } } \ =$ 0.2.

# 7.3 Pre-Processing Phase

During the offline pre-preprocessing phase, we find CS and RS for data lake tables using an external KB (Sec. 4), and discover FDs in the data lake tables to create a synthesized KB (Sec. 7.2). To reduce query time, we create two inverted indexes. First, a node inverted index maps a column to its CS with respective $C S _ { \mathrm { C o N F } }$ . The second is an edge inverted index that maps RS to its connected columns in the same table, with $R S _ { \mathrm { C o v F } }$ .

Time Complexity. Consider a set of data lake tables $\mathcal { T }$ and let $m$ and $n$ be the largest number of columns and rows respectively in a data lake table. We make a linear pass over each column and record the count of each candidate CS (include candidate top-level types) in an inverted index. So, the asymptotic time complexity of computing CS is $O ( | \mathcal { T } | \cdot m \cdot n )$ . Similarly, for computing RS, we only consider columns with non-empty CS, which can then find RS in the KB [36]. Let $m _ { c }$ be the largest number of columns having CS in a data lake table such that $m _ { c } \le m$ . The time taken to compute RS is bounded by $O ( | \mathcal { T } | \cdot m _ { c } ^ { 2 } \cdot n )$ .

The time complexity of creating synthesized KB includes the creation of Synthesized Type Dictionary and Synthesized Relationship Dictionary. Synthesized Type Dictionary creation is analogous to CS computation, where we make a linear pass over each column $( O ( | { \mathcal { T } } | \cdot m \cdot n )$ time). Also, the creation of Synthesized Relationship Dictionary is analogous to RS computation. But in addition, we also need to mine unary FDs, which is quadratic in the number of columns [12]. So, its creation is linear in the highest number of rows and quadratic only in the highest number of columns across

any table i.e., $O ( | \mathcal { T } | \cdot m ^ { 2 } \cdot n )$ . Recall that this is a pre-processing task and does not need to be done during the query phase. We report the pre-processing time on different benchmarks in Sec. 8.4.

# 7.4 Query Phase

At query time, shown in Fig. 4(b), the user provides a query table ?? in which the intent column is marked.10 We then create the semantic tree for $Q$ , rooted at the intent column. For this, we access KB and synthesized KB to find the CS and RS for the query table. Given the annotated semantics of the query table, SANTOS searches for unionable tables. It does so by retrieving a set of candidate tables and their respective confidence scores from the inverted index. SANTOS computes the unionability score between column pairs of ?? and candidate tables based on different hierarchy levels, allowing us to match columns of different granularities (see Eq. (7)).

Time complexity. Given a query table $Q$ with m columns and n rows, we compute its CS from both KB and synthesized KB by making a single pass over each column, which takes $O ( m \cdot n )$ time. Recall that we only find RS for columns that have non-empty CS. Let $m _ { c }$ be the number of columns having CS such that $m _ { c } \le m$ . RS can thus be computed in $O ( m _ { c } ^ { 2 } . n )$ time. After finding CS and RS for the query table, the computation of unionability score depends linearly on the number of semantics found for the query table. With inverted indexes and the number of possible CS and RS as constant during the query phase, it is efficient to compute the unionability score. We report the query time using real data lake tables in Sec. 8.4.

# 8 EXPERIMENTS

In this section, we compare SANTOS with a state-of-the-art unionability approach $D ^ { 3 } L$ [3] that uses column unionability. Notice that $D ^ { 3 } L$ builds on Table Union Search [33] by adding regular expressions, domain distributions, etc. Hence, we only compare with $D ^ { 3 } L$ to analyze the importance of relationship semantics in union search.

# 8.1 Experimental Setup

SANTOS is implemented in Python on a server with Intel(R) Xeon(R) Gold 5218 CPU @ 2.30GHz processor. Our code is publicly available.11 Our experiments aim to answer the following questions:

(1) How effective is SANTOS in returning top- $k$ unionable tables relative to the baseline given a query table? (Sec. 8.2)   
(2) How do each component of SANTOS (use of existing KB, use of synthesized KB, and use of both) influence the quality of the results? (Sec. 8.3)   
(3) How well does SANTOS scale over real data lakes, as compared to prior work? (Sec. 8.4)

Evaluation Measures. Since our method, along with other table union search methods, returns a top- $k$ list of unionable tables, we use mean average precision $( M A P @ k )$ to evaluate the effectiveness of table union search approaches [27]. Following previous works [3, 33], we also report Precision at $k$ (??@??) and Recall at $k$ $( R @ k )$ [27]. When creating our ground truth for evaluations, we assign a binary score $\in \{ 0 , 1 \}$ to a data lake table ?? , given a query table $Q$ to label $T$ as unionable or not-unionable to ??. Formally, let $\mathcal { T } _ { Q }$ be the set

of unionable tables based on the ground truth and $\hat { \mathcal { T } } _ { Q }$ be the set of top- $k$ unionable tables based on some method with respect to a query table $Q$ . Then, $P ( \varpi k$ and $R @ k$ with respect to $Q$ are given by:

$$
P @ k = \frac {\mathcal {T} _ {Q} \cap \hat {\mathcal {T}} _ {Q}}{\hat {\mathcal {T}} _ {Q}}, R @ k = \frac {\mathcal {T} _ {Q} \cap \hat {\mathcal {T}} _ {Q}}{\mathcal {T} _ {Q}} \tag {12}
$$

Note that the size of $\hat { \mathcal { T } } _ { Q }$ is set to $k$ , while $\mathcal { T } _ { Q }$ may, in general, be larger (or smaller) than $k$ . To best understand the results, we create benchmarks where the ground truth $( \mathcal { T } _ { Q } )$ is at least $k$ . Hence, using $P ( \varpi k$ , if a method returns less than $k$ results, the results not returned are considered incorrect (false). For instance, if $k = 1 0$ and the ground truth has 20 results, if a method returns only 2 results out of which 1 is correct and the other is incorrect, then $\begin{array} { r } { P @ 1 0 = \frac { 1 } { 1 0 } } \end{array}$ and $\begin{array} { r } { R @ 1 0 = \frac { 1 } { 2 0 } } \end{array}$ . So when $k < | \mathcal { T } _ { Q } |$ then perfect recall is not possible as the $R @ k$ can at best be $\frac { k } { | \mathcal { T } _ { Q } | }$ .

The mean average precision $( M A P @ k )$ , defined as follows:

$$
M A P @ k = \frac {1}{\left| \hat {\mathcal {T}} _ {Q} \right|} \sum_ {k = 1} ^ {\left| \hat {\mathcal {T}} _ {Q} \right|} P @ k \tag {13}
$$

We compute $P ( \omega k , R @ k$ , and $M A P @ k$ for each query and report the average performance over several queries for a fixed $k$ (e.g. average $P ( \varpi k )$ ). We also measure the pre-processing and query times for scalability experiments.

Benchmarks. Figure 6 details the statistics of the benchmarks that we use, both for their data lake tables and the query tables. The benchmarks are publicly available. 12

Figure 6: Benchmarks used in the experiments.   

<table><tr><td rowspan="2">Table Source</td><td colspan="3">Data lake Tables</td><td colspan="3">Query Tables</td></tr><tr><td># Tables</td><td># Columns</td><td># Rows</td><td># Tables</td><td># Columns</td><td># Rows</td></tr><tr><td>TUS</td><td>1,530</td><td>14,810</td><td>6.8 M</td><td>125</td><td>1,610</td><td>557 K</td></tr><tr><td>SMALL</td><td>550</td><td>6,322</td><td>3.8 M</td><td>50</td><td>615</td><td>1.07 M</td></tr><tr><td>LARGE</td><td>11,090</td><td>123,477</td><td>70 M</td><td>80</td><td>1,017</td><td>1.03 M</td></tr></table>

1. TUS Benchmark (TUS): The TUS benchmark [33] focuses on attribute unionability, ignoring relationships between columns. Specifically, two tables in the benchmark are unionable if they have unionable attributes. To repurpose this benchmark, we labeled tables in the benchmark unionable if they share relationships, not just attributes. Specifically, out of the 10 original seed tables used to produce 1530 tables in the benchmark, we found meaningful relationships in tables that originate from 6 seed tables. From these, we randomly selected 125 tables as Query Tables and marked the intent columns. We use the original data lake from the benchmark.   
2. SANTOS Small Benchmark (SMALL): The SANTOS Small Benchmark contains 550 data lake tables from Canada, UK, US, and Australian open data. First, we collected 296 real data lake tables from 35 distinct domains. To further expand the benchmark, we selected 19 large tables among them and manually annotated the relationships between their columns. Using the benchmark creation technique from TUS [33], we then partitioned the annotated tables horizontally and vertically to obtain 254 non-overlapping tables, thereby increasing the total number of tables to 550. Then we randomly selected 50 tables having at least 10 unionable tables

as query tables (at most 2 query tables from each domain) and labeled the intent columns and the ground truth. As the tables are taken from real data lakes, they contain nulls, string values, date, numerical values, etc. We will refer to this benchmark as SMALL.

3. SANTOS Large Benchmark (LARGE): To evaluate SANTOS in a broader environment, we collected 11,090 real tables from Canada and UK Open Data for the data lake. From these tables, we randomly selected 80 tables as query tables, each having at least 20 unionable tables, and marked their intent columns. For this benchmark, we only report $P _ { \ @ k }$ and $M A P @ k$ (and runtimes) as reporting $R @ k$ requires a laborious annotation of the full data lake. For similar reasoning, we manually verify $P ( \varpi k$ and $M A P @ k$ only up to $k = 2 0$ Like SMALL Benchmark, these tables also contain nulls, string values, dates, numerical values, etc. We refer to this benchmark as LARGE from now on.

Baselines. The table union search problem based on determining if column values are drawn from the same domain was first defined and addressed by Nargesian et al. [33]. Recently, Bogatu et al. proposed $D ^ { 3 } L$ for the broader problem of finding related tables (both joinable and unionable tables) [3]. $D ^ { 3 } L$ adds metrics based on column names, regular expressions and domain distributions to the word-embedding and value overlap-based models of Nargesian et al. [33]. Therefore, we compare SANTOS to $D ^ { 3 } L$ with these extended metrics, by reproducing $D ^ { 3 } L$ using their code.13

TURL is a recent method that uses representational learning over web tables [8]. TURL learns table representations that successfully find CS (column type annotation) and RS (relation extraction) in web tables [8]. Although it does not support union search directly, we extended it to create a SANTOS-like technique. Specifically, we treat TURL as a KB and, similar to SANTOS, use it to annotate the CS and RS for each table. Then we index the data lake tables similarly to the method in Sec. 4. This approach provides an analysis of a learning-based alternative that uses a pre-trained model.

# 8.2 SANTOS Effectiveness vs. Baselines

To analyze the effectiveness of SANTOS, we compare ??@??, ??@?? and $M A P @ k$ of SANTOS against our baselines $D ^ { 3 } L$ and TURL. We then analyze variations of SANTOS in an ablation study in Sec. 8.3.

Figure 7 reports the average performance of SANTOS compared to the baseline $D ^ { 3 } L$ on all three benchmarks. We only report TURL for TUS benchmark, since its performance is similar on the other benchmarks, with its measures significantly less than SANTOS. For consistency with previous research [3, 33], and to ensure that each query table has at least $k$ expected unionable tables, we report results for $k = 6 0$ on TUS. Considering the number of unionable tables per query table, we report results at $k = 1 0$ on SMALL and at $k = 2 0$ on LARGE (Fig. 7). Note that when the ground truth contains more than $k$ results, $1 0 0 \% R @ k$ is not possible. Rather the highest possible $R @ 6 0$ is around $6 2 \%$ for TUS and $R @ 1 0$ is around $7 2 \%$ for SMALL (Ideal lines in (b) and (d)). In the LARGE data lake the whole corpus is not labeled. Therefore, we cannot report $R @ k$

We first evaluated a version of SANTOS that only considers column semantics, $\mathrm { S A N T O S } _ { \mathrm { C o l } }$ , compared against $D ^ { 3 } L$ . On TUS, $\mathrm { S A N T O S } _ { \mathrm { C o l } }$ has $M A P @ 6 0$ of $6 5 \%$ and $P @ 6 0$ of $6 2 \%$ , which is comparable to $D ^ { 3 } L$ results. When we included relationship semantics

<table><tr><td>Benchmark</td><td>Method</td><td>MAP@k</td><td>P@k</td><td>R@k</td></tr><tr><td rowspan="3">TUS(k=60)</td><td>TURL</td><td>0.13</td><td>0.16</td><td>0.08</td></tr><tr><td>D3L</td><td>0.64</td><td>0.58</td><td>0.31</td></tr><tr><td>SANTOS</td><td>0.80</td><td>0.70</td><td>0.37</td></tr><tr><td rowspan="2">SMALL(k=10)</td><td>D3L</td><td>0.52</td><td>0.58</td><td>0.42</td></tr><tr><td>SANTOS</td><td>0.93</td><td>0.90</td><td>0.68</td></tr><tr><td rowspan="2">LARGE(k=20)</td><td>D3L</td><td>0.29</td><td>0.26</td><td>-</td></tr><tr><td>SANTOS</td><td>0.77</td><td>0.73</td><td>-</td></tr></table>

![](images/a440cd19d3157ac3ac8c8f19da7e14bd61f662506bda023ba25d471d1bc16f7c.jpg)  
Figure 7: Comparison of P@k, MAP@k and $\mathbf { R } @ \mathbf { k }$ of TURL, $D ^ { 3 } L$ and $\mathrm { S A N T O S } _ { \mathrm { F u l l } }$ on TUS, and $D ^ { 3 } L$ and $\mathrm { S A N T O S } _ { \mathrm { F u l l } }$ on SMALL and LARGE benchmarks.

![](images/51e5fa80441a5026c78fcdd6e38e6552033b864ffe4c35656a32c38cc374b7bf.jpg)  
(a) Average ??@?? on TUS   
(c) Average ??@?? on SMALL

![](images/edbbc7071434e8bc5a2a860b512c5d53d1f652ee7bdd1b6c382b6fb505c23563.jpg)  
(d) Average ??@?? on SMALL   
Figure 8: Effectiveness of SANTOS and its variations against baselines in different benchmarks

and evaluated full SANTOS against $D ^ { 3 } L$ , however, SANTOS performs better than $D ^ { 3 } L$ over all measures in all benchmarks. On TUS, SANTOS outperforms $D ^ { 3 } L$ by over $2 0 \%$ , $2 5 \%$ and $1 9 \%$ in terms of $P @ 6 0$ , $M A P @ 6 0$ , and $R @ 6 0$ , respectively. On SMALL, SANTOS outperforms $D ^ { 3 } L$ by over $5 6 \%$ , $7 8 \%$ , and $6 1 \%$ for $P @ 1 0$ , ??????@10, and $R @ 1 0$ respectively. In terms of $R { \textcircled { a } } k$ , SANTOS is closer to ideal recall than the baseline in both benchmarks. This indicates that relationship semantics is important in union search.

On a real data lake (LARGE), we observe even further improvements ( $P @ 2 0$ and $M A P @ 2 0$ by over $1 8 0 \%$ and $1 6 5 \%$ compared to $D ^ { 3 } L$ respectively), indicating that relationships are even more important in this benchmark. For illustration, consider a query table $Q$ in LARGE benchmark about biodiversity in different counties (with columns like county_name, animal_scientific_name, documented_year, etc.). For this table, the top- $k$ results by SAN-TOS contain tables about alpine birds, fish, and trees and the places they are found, which seems to be correct. However, as discussed in Ex. 1, although tables returned by $D ^ { 3 } L$ have common (unionable) columns, they are about different topics with different relationships.

![](images/55d3aacbfa0f94092478408eb8594951ffccb344c9b2544f3800088d3bf10e99.jpg)  
(a) MAP@60 on TUS

![](images/100a73c92c8b2ad371cf784d678dc7efd02136182dcd04958c2657475b3bdb5c.jpg)  
(b) MAP@10 on SMALL   
Figure 9: Average ??????@?? of $\mathrm { S A N T O S } _ { \mathrm { F u l l } }$ ( $k = 6 0$ on TUS, $k =$ 10 on SMALL) for different percentages of the existing KB

For instance, they include tables about emergency hospital admissions after accidents because they contain columns like county (unionable with county_name in ??) that describes the place of accident and year (unionable with documented_year) that describes when the accident took place. For both approaches, we manually verify $P ( \varpi k$ and $M A P @ k$ up to $k = 2 0$ . The raw results for each query table on LARGE by SANTOS and $D ^ { 3 } L$ are available in the supplementary materials. 12

Figure 8 shows detailed comparison of precision and recall for different values of $k$ in both TUS and SMALL benchmarks. In these graphs, SANTOS is labeled $\mathrm { S A N T O S } _ { \mathrm { F u l l } }$ . The other lines labeled SANTOS will be explained below. For each $k$ , SANTOS outperforms competing methods. Notice that perfect $R { \textcircled { a } } k$ is also plotted in Fig. 8(b) (solid black line). TURL-based implementation has the least precision $P (  \omega 6 0 = 0 . 1 5 )$ , MAP $M A P @ 6 0 = 0 . 1 _ { \cdot }$ ) and Recall $( R @ 6 0 ~ = ~ 0 . 1 )$ ) on TUS. It is possible that the reason TURL performs poorly is that is trained over web tables, which have different characteristics from real open data [29]. We only report TURL performance on TUS as the results in other benchmarks show a similar trend that is well below $D ^ { 3 } L$ and SANTOS.

# 8.3 SANTOS Effectiveness Ablation Study

We perform an ablation study to understand the impact of one of our key innovations, the use of a synthesized KB, on the accuracy of SANTOS. We compare the full version of SANTOS $( { \mathrm { S A N T O S } } _ { \mathrm { F u l l } } )$ ), a version of using solely the existing KB $S A N T O S _ { K B }$ ), and a version only using the synthesized KB (SANTOSSynth) over varying values of $k$ on TUS and SMALL benchmarks. Fig. 8 shows these comparisons in detail. $\mathrm { S A N T O S } _ { \mathrm { F u l l } }$ obtains the best results over all values of $k$ in both benchmarks. On TUS (Fig. 8 (a) and (b)), SANTOSSynth (circle/grey line) has lower $P ( \varpi k$ and $R @ k$ than $\operatorname { S A N T O S } _ { \mathrm { K B } }$ because it is not able to retrieve enough results for all the query tables. However, using the synthesized KB with the existing KB together $( { \mathrm { S A N T O S } } _ { \mathrm { F u l l } } )$ ) provides the best performance. We see a different trend on SMALL (Fig. 8 (c) and (d)) where $S A N T O S _ { \mathrm { K B } }$ was not able to return enough results. Specifically, the existing KB had no coverage for 14 of the 50 query tables. However, $\mathsf { S A N T O S } _ { \mathrm { S y n t h } }$ was able to handle those queries and hence, maintain the overall performance of $\mathrm { S A N T O S } _ { \mathrm { F u l l } }$ . This suggests that the synthesized KB helps SANTOS alleviate the effect from the imperfect coverage of a KB. Note that we ran our experiments using an open KB over open data tables. YAGO may cover less (or more) entities in an enterprise data lake, but enterprises generally maintain their own domain-specific

KB’s [18, 44, 45]. SANTOS can easily be adapted to such data lakes by augmenting YAGO with the respective enterprise KBs.

To better understand the contribution of the KB and synthesized KB, we compute the average $M A P @ k$ of $\mathrm { S A N T O S } _ { \mathrm { F u l l } }$ by varying the percentage of information from the existing KB randomly. We systematically remove portions of the existing KB entities that are in the data lake tables and evaluate how the synthesized KB compensates for the loss of KB coverage. Fig. 9 shows this analysis on the TUS and SMALL benchmarks for $k = 6 0$ and $k = 1 0$ , respectively. We first turn off the existing KB and compute $M A P @ k$ for SANTOSSynth. Then we gradually increase the existing KB coverage until we reach $1 0 0 \%$ . In both benchmarks, increasing the usage of the KB increases SANTOS’s effectiveness almost linearly, empirically validating that SANTOS performs better with more KB coverage. Furthermore, it shows the significance of both $\mathrm { S A N T O S _ { K B } }$ and $\mathsf { S A N T O S } _ { \mathrm { S y n t h } }$ where, $S A N T O S _ { K B }$ increases $\mathrm { S A N T O S } _ { \mathrm { F u l l } }$ ’s ?????? by $1 8 \%$ on TUS and $1 5 \%$ on SMALL and $\mathsf { S A N T O S } _ { \mathrm { S y n t h } }$ increases it by $8 \%$ on TUS and $4 3 \%$ on SMALL. This shows that SANTOSSynth alone has decent accuracy ( $6 8 \%$ on TUS and $8 1 \%$ on SMALL), and incrementally adding entries from the KB improves the accuracy further in a near-linear trend. Thus, leveraging relationship semantics for union search is benefited by the use of both KB’s.

Recall that, SANTOS only uses cell values to create the semantic graphs. So, the accuracy of the created graph may depend on the number of rows in the query tables i.e., fewer rows may impact its accuracy. However, even a human expert would need an adequate number of rows to understand the table semantics and so does SANTOS. As reported in Fig. 6, the data lake tables generally contain thousands of rows and hence, SANTOS is fairly effective in understanding their semantics as reported in our experiments.

# 8.4 SANTOS Scalability

Figure 10: Comparison of Indexing and Query times of SANTOSSynth, ${ \mathrm { S A N T O S } } _ { \mathrm { K B } }$ , $\mathrm { S A N T O S } _ { \mathrm { F u l l } }$ , and $D ^ { 3 } L$ in different benchmarks. For Query time, we report both the average query time and in parenthesis we include ${ \bf 8 0 \% }$ confidence interval ( $\mathbf { 1 0 \% }$ to $9 0 \%$ percentile).   

<table><tr><td>Benchmark</td><td>Method</td><td>Indexing</td><td>Query (sec)</td></tr><tr><td rowspan="4">TUS</td><td>D3L</td><td>1 hr 21 min</td><td>54.1 (20.5 – 97.3)</td></tr><tr><td>SANTOSFull</td><td>4 hr 26 min</td><td>22.9 (1.7 – 48.6)</td></tr><tr><td>SANTOSKB</td><td>1 hr 38 min</td><td>6.1 (0.7 – 13.9)</td></tr><tr><td>SANTOSynth</td><td>3 hr 45 min</td><td>15.6 (0.7 – 43.2)</td></tr><tr><td rowspan="4">SMALL</td><td>D3L</td><td>17 min</td><td>22.4 (7.4 – 43.3)</td></tr><tr><td>SANTOSFull</td><td>4 hr 46 min</td><td>28.2 (0.8 – 102)</td></tr><tr><td>SANTOSKB</td><td>1 hr 8 min</td><td>10.0 (0.3 – 33.6)</td></tr><tr><td>SANTOSynth</td><td>3 hr 41 min</td><td>18.2 (0.5 – 98.6)</td></tr><tr><td rowspan="2">LARGE</td><td>D3L</td><td>7 hr 7 min</td><td>177 (13.0 – 325.0)</td></tr><tr><td>SANTOSFull</td><td>21 hr 59 min</td><td>35.8 (0.21 – 57.2)</td></tr></table>

As a final analysis, we experimentally validate that SAN-TOS scales to large data lakes. We report indexing and query times for SANTOS $( { \mathrm { S A N T O S } } _ { \mathrm { F u l l } } )$ ), the two variations of SANTOS $( \mathrm { S A N T O S _ { K B } }$ and SANTOSSynth) and $D ^ { 3 } L$ on all three benchmarks (Fig. 10). For discussion, we focus on LARGE as it is largest in size.

Although $D ^ { 3 } L$ is around 3x faster than SANTOS when indexing data lake tables (Sec. 7.3), the relationship-based approach of SANTOS proves to be significantly more effective than the columnbased approach of $D ^ { 3 } L$ . Nevertheless, SANTOS is much faster than $D ^ { 3 } L$ at query time (Sec. 7.4). In LARGE, SANTOS takes 7 sec on average to index each data lake table, and thus is able to scale to tables with thousands of rows and columns (see Fig. 6). The individual index creation time of $\mathrm { S A N T O S } _ { \mathrm { S y n t h } }$ , which includes the time taken to discover FDs, is $1 7 \mathrm { { h r } \ 1 9 \mathrm { { m i n } } }$ . Recall that SANTOS may be further effective if the tables do not contain homographs. Using the state-of-the-art technique [22], the time taken to detect the homographs is 17 min on the TUS Benchmark. Therefore, we can detect homographs in fairly small and feasible pre-processing time. Notice however, the absence of homographs is not a necessary condition and SANTOS is still more effective than baselines without this pre-processing step (see Sec. 8.2). Similarly, SANTOSKB’s index creation time is $1 2 \mathrm { h r } 4 0 \mathrm { m i n }$ . One can create these indexes in parallel, so $\mathrm { S A N T O S } _ { \mathrm { F u l l } }$ indexing time can alternatively be max(17 hr 19 min, $1 2 \mathrm { { h r } 4 0 \mathrm { { m i n } ) = 1 7 } }$ hr 19 min plus few seconds to combine the indexes (rather than $2 1 \mathrm { h r } 5 9 \mathrm { m i n }$ reported in Fig. 10). The synthesized KB created over LARGE uses 3588 MB of main memory at query time and occupies 3441 MB in the secondary storage. 9 This further validates SANTOS’s scalability. In the future work, we will study other spaces for optimizing the creation of synthesized KB.

We analyze 125 Query Tables on TUS, 50 Query Tables on SMALL and 80 query tables on LARGE, for which we report the average, 10th, and 90th percentile query times. Notice that the query time can vary depending on the complexity of the query table. SANTOS average query times are faster than $D ^ { 3 } L$ query time, since $D ^ { 3 } L$ searches through each of their five indexes when finding unionable tables, whereas we only search through synthesized KB index, YAGO index, or both. Also, the query time of $D ^ { 3 } L$ and SANTOS are comparable on SMALL because $D ^ { 3 } L$ is faster on smaller data lakes. However, SANTOS is faster than $D ^ { 3 } L$ for larger data lakes as suggested by the average query time on TUS (over 3 times faster) and LARGE (almost 6 times faster). All in all, even with the new synthesized KB, our indexing and query times are comparable or even faster than those of a state-of-the-art approach.

# 9 CONCLUSION

We presented SANTOS, a method for finding unionable tables in data lakes based on both column and relationship semantics. SAN-TOS discovers and uses relationship semantics between pairs of columns in a table using an existing knowledge base (KB) and a synthesized KB created by exploiting the knowledge of the data lake. We conducted experiments on an adapted version of the existing TUS benchmark as well as our new SMALL and LARGE benchmarks and showed that SANTOS unionability search outperforms a state-of-the-art table union approach. Also, the experimental results showed the robustness of our approach and the importance of our synthesized KB in overcoming curated KBs with limited coverage.

Acknowledgements. This work was supported in part by NSF under award numbers IIS-1956096, IIS-2107248 and IIS-1762268.

# REFERENCES

[1] Marco D. Adelfio and Hanan Samet. 2013. Schema Extraction for Tabular Data on the Web. Proc. VLDB Endow. 6, 6 (apr 2013), 421–432. https://doi.org/10.14778/ 2536336.2536343   
[2] Akiko Aizawa. 2003. An information-theoretic perspective of tf–idf measures. Information Processing and Management 39, 1 (2003), 45–65. https://doi.org/10. 1016/S0306-4573(02)00021-3   
[3] Alex Bogatu, Alvaro A. A. Fernandes, Norman W. Paton, and Nikolaos Konstantinou. 2020. Dataset Discovery in Data Lakes. In ICDE 2020. IEEE, 709–720. https://doi.org/10.1109/ICDE48307.2020.00067   
[4] Dan Brickley, Matthew Burgess, and Natasha F. Noy. 2019. Google Dataset Search: Building a search engine for datasets in an open Web ecosystem. In WWW 2019. ACM, 1365–1375. https://doi.org/10.1145/3308558.3313685   
[5] Michael J. Cafarella, Alon Y. Halevy, and Nodira Khoussainova. 2009. Data Integration for the Relational Web. Proc. VLDB Endow. 2, 1 (aug 2009), 1090–1101. https://doi.org/10.14778/1687627.1687750   
[6] Riccardo Cappuzzo, Paolo Papotti, and Saravanan Thirumuruganathan. 2020. Creating Embeddings of Heterogeneous Relational Datasets for Data Integration Tasks. In SIGMOD 2020. Association for Computing Machinery, 1335–1349. https: //doi.org/10.1145/3318464.3389742   
[7] Anish Das Sarma, Lujun Fang, Nitin Gupta, Alon Halevy, Hongrae Lee, Fei Wu, Reynold Xin, and Cong Yu. 2012. Finding Related Tables. In SIGMOD 2012. Association for Computing Machinery, 817–828. https://doi.org/10.1145/2213836. 2213962   
[8] Xiang Deng, Huan Sun, Alyssa Lees, You Wu, and Cong Yu. 2020. TURL: Table Understanding through Representation Learning. Proc. VLDB Endow. 14, 3 (nov 2020), 307–319. https://doi.org/10.14778/3430915.3430921   
[9] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1. ACL, 4171–4186. https://doi.org/10.18653/v1/N19-1423   
[10] Mina Farid, Alexandra Roatis, Ihab F. Ilyas, Hella-Franziska Hoffmann, and Xu Chu. 2016. CLAMS: Bringing Quality to Data Lakes. In SIGMOD 2016. Association for Computing Machinery, 2089–2092. https://doi.org/10.1145/2882903.2899391   
[11] Raul Castro Fernandez, Essam Mansour, Abdulhakim Ali Qahtan, Ahmed K. Elmagarmid, Ihab F. Ilyas, Samuel Madden, Mourad Ouzzani, Michael Stonebraker, and Nan Tang. 2018. Seeping Semantics: Linking Datasets Using Word Embeddings for Data Discovery. ICDE (2018), 989–1000. https://doi.org/10.1109/ICDE.2018.00093   
[12] Peter A. Flach and Iztok Savnik. 1999. Database Dependency Discovery: A Machine Learning Approach. AI Commun. 12, 3 (1999), 139–160. http://content. iospress.com/articles/ai-communications/aic182   
[13] D Frank Hsu and Isak Taksa. 2005. Comparing rank and score combination methods for data fusion in information retrieval. Information retrieval 8, 3 (2005), 449–480. https://doi.org/10.1007/s10791-005-6994-4   
[14] Sainyam Galhotra and Udayan Khurana. 2020. Semantic Search over Structured Data. In CIKM 2020. Association for Computing Machinery, 3381–3384. https: //doi.org/10.1145/3340531.3417426   
[15] Parantapa Goswami, Eric Gaussier, and Massih-Reza Amini. 2017. Exploring the space of information retrieval term scoring functions. Information Processing and Management 53, 2 (2017), 454–472. https://doi.org/10.1016/j.ipm.2016.11.003   
[16] Vinh Thinh Ho, Yusra Ibrahim, Koninika Pal, Klaus Berberich, and Gerhard Weikum. 2019. Qsearch: Answering Quantity Queries from Text. In ISWC. Springer-Verlag, 237–257. https://doi.org/10.1007/978-3-030-30793-6_14   
[17] Vinh Thinh Ho, Koninika Pal, and Gerhard Weikum. 2021. QuTE: Answering Quantity Queries from Web Tables. In SIGMOD (Virtual Event, China). Association for Computing Machinery, New York, NY, USA, 2740–2744. https: //doi.org/10.1145/3448016.3452763   
[18] Aidan Hogan, Eva Blomqvist, Michael Cochez, Claudia D’amato, Gerard De Melo, Claudio Gutierrez, Sabrina Kirrane, José Emilio Labra Gayo, Roberto Navigli, Sebastian Neumaier, Axel-Cyrille Ngonga Ngomo, Axel Polleres, Sabbir M. Rashid, Anisa Rula, Lukas Schmelzeisen, Juan Sequeda, Steffen Staab, and Antoine Zimmermann. 2021. Knowledge Graphs. ACM Comput. Surv. 54, 4, Article 71 (July 2021), 37 pages. https://doi.org/10.1145/3447772   
[19] Madelon Hulsebos, Kevin Hu, Michiel Bakker, Emanuel Zgraggen, Arvind Satyanarayan, Tim Kraska, Çagatay Demiralp, and César Hidalgo. 2019. Sherlock: A Deep Learning Approach to Semantic Data Type Detection. In KDD 2019. ACM, 1500–1508. https://doi.org/10.1145/3292500.3330993   
[20] Oliver Lehmberg and Christian Bizer. 2017. Stitching Web Tables for Improving Matching Quality. Proc. VLDB Endow. 10, 11 (Aug. 2017), 1502–1513. https: //doi.org/10.14778/3137628.3137657   
[21] Oliver Lehmberg, Dominique Ritze, Robert Meusel, and Christian Bizer. 2016. A Large Public Corpus of Web Tables Containing Time and Context Metadata. In WWW 2016 Companion. International World Wide Web Conferences Steering Committee, 75–76. https://doi.org/10.1145/2872518.2889386   
[22] Aristotelis Leventidis, Laura Di Rocco, Wolfgang Gatterbauer, Renée J. Miller, and Mirek Riedewald. 2021. DomainNet: Homograph Detection for Data Lake

Disambiguation. In EDBT 2021. OpenProceedings.org, 13–24. https://doi.org/10. 5441/002/edbt.2021.03   
[23] Girija Limaye, Sunita Sarawagi, and Soumen Chakrabarti. 2010. Annotating and Searching Web Tables Using Entities, Types and Relationships. Proc. VLDB Endow. 3, 1–2 (Sept. 2010), 1338–1347. https://doi.org/10.14778/1920841.1921005   
[24] Xiao Ling, Alon Y. Halevy, Fei Wu, and Cong Yu. 2013. Synthesizing Union Tables from the Web. In IJCAI 2013. IJCAI/AAAI, 2677–2683. http://www.aaai.org/ocs/ index.php/IJCAI/IJCAI13/paper/view/6758   
[25] H. P. Luhn. 1957. A Statistical Approach to Mechanized Encoding and Searching of Literary Information. IBM Journal of Research and Development 1, 4 (1957), 309–317. https://doi.org/10.1147/rd.14.0309   
[26] David JC MacKay. 2003. Information theory, inference and learning algorithms. Cambridge university press. https://books.google.com/books?id=AKuMj4PN_ EMC   
[27] C.D. Manning, P. Raghavan, and H. Schütze. 2008. Introduction to Information Retrieval. Cambridge University Press. https://books.google.com/books?id= t1PoSh4uwVcC   
[28] Suvodeep Mazumdar and Ziqi Zhang. 2016. Visualizing Semantic Table Annotations with TableMiner+. In ISWC 2016, Vol. 1690. CEUR-WS.org. http://ceurws.org/Vol-1690/paper88.pdf   
[29] Renée J. Miller. 2018. Open Data Integration. Proc. VLDB Endow. 11, 12 (aug 2018), 2130–2139. https://doi.org/10.14778/3229863.3240491   
[30] Renée J. Miller, Fatemeh Nargesian, Erkang Zhu, Christina Christodoulakis, Ken Q. Pu, and Periklis Andritsos. 2018. Making Open Data Transparent: Data Discovery on Open Data. IEEE Data Eng. Bull. 41, 2 (2018), 59–70. http://sites.computer. org/debull/A18june/p59.pdf   
[31] Varish Mulwad, Tim Finin, Zareen Syed, and Joshi Anupam. 2010. Using linked data to interpret tables. In Proceedings of the the First International Workshop on Consuming Linked Data, Vol. 665. CEUR-WS.org. http://ceur-ws.org/Vol-665/MulwadEtAl_COLD2010.pdf   
[32] Fatemeh Nargesian, Erkang Zhu, Renée J. Miller, Ken Q. Pu, and Patricia C. Arocena. 2019. Data Lake Management: Challenges and Opportunities. Proc. VLDB Endow. 12, 12 (aug 2019), 1986–1989. https://doi.org/10.14778/3352063. 3352116   
[33] Fatemeh Nargesian, Erkang Zhu, Ken Q. Pu, and Renée J. Miller. 2018. Table Union Search on Open Data. Proc. VLDB Endow. 11, 7 (March 2018), 813–825. https://doi.org/10.14778/3192965.3192973   
[34] Masayo Ota, Heiko Müller, Juliana Freire, and Divesh Srivastava. 2020. Data-Driven Domain Discovery for Structured Datasets. Proc. VLDB Endow. 13, 7 (March 2020), 953–967. https://doi.org/10.14778/3384345.3384346   
[35] Christos H Papadimitriou, Prabhakar Raghavan, Hisao Tamaki, and Santosh Vempala. 2000. Latent semantic indexing: A probabilistic analysis. J. Comput. System Sci. 61, 2 (2000), 217–235. https://doi.org/10.1006/jcss.2000.1711   
[36] Thomas Pellissier Tanon, Gerhard Weikum, and Fabian Suchanek. 2020. YAGO 4: A Reason-able Knowledge Base. In ESWC. Springer International Publishing, 583–596. https://doi.org/10.1007/978-3-030-49461-2_34   
[37] Juan Ramos et al. 2003. Using tf-idf to determine word relevance in document queries. In Proceedings of the first instructional conference on machine learning, Vol. 242. Citeseer, 29–48.   
[38] Dominique Ritze, Oliver Lehmberg, and Christian Bizer. 2015. Matching HTML Tables to DBpedia. In WIMS 2015. Association for Computing Machinery, Article 10, 6 pages. https://doi.org/10.1145/2797115.2797118   
[39] Karen Sparck Jones. 1988. A Statistical Interpretation of Term Specificity and Its Application in Retrieval. Taylor Graham Publishing, GBR, 132–142.   
[40] Yoshihiko Suhara, Jinfeng Li, Yuliang Li, Dan Zhang, Çağatay Demiralp, Chen Chen, and Wang-Chiew Tan. 2022. Annotating Columns with Pre-Trained Language Models. In SIGMOD 2022. Association for Computing Machinery, 1493–1503. https://doi.org/10.1145/3514221.3517906   
[41] Zareen Syed, Tim Finin, Varish Mulwad, and Anupam Joshi. 2010. Exploiting a Web of Semantic Data for Interpreting Tables. In Proceedings of the Second Web Science Conference. ACM. https://ebiquity.umbc.edu/paper/html/id/474   
[42] Kunihiro Takeoka, Masafumi Oyamada, Shinji Nakadai, and Takeshi Okadome. 2019. Meimei: An Efficient Probabilistic Approach for Semantically Annotating Tables. In AAAI 2019. AAAI Press, 281–288. https://doi.org/10.1609/aaai.v33i01. 3301281   
[43] Petros Venetis, Alon Halevy, Jayant Madhavan, Marius Paşca, Warren Shen, Fei Wu, Gengxin Miao, and Chung Wu. 2011. Recovering Semantics of Tables on the Web. Proc. VLDB Endow. 4, 9 (June 2011), 528–538. https://doi.org/10.14778/ 2002938.2002939   
[44] Muhammad Yahya, John G. Breslin, and Muhammad Intizar Ali. 2021. Semantic Web and Knowledge Graphs for Industry 4.0. Applied Sciences 11, 11 (2021). https://doi.org/10.3390/app11115110   
[45] Nasser Zalmout, Chenwei Zhang, Xian Li, Yan Liang, and Xin Luna Dong. 2021. All You Need to Know to Build a Product Knowledge Graph. In KDD 2021. Association for Computing Machinery, 4090–4091. https://doi.org/10.1145/3447548. 3470825   
[46] Dan Zhang, Madelon Hulsebos, Yoshihiko Suhara, Çağatay Demiralp, Jinfeng Li, and Wang-Chiew Tan. 2020. Sato: Contextual Semantic Type Detection in

Tables. Proc. VLDB Endow. 13, 12 (2020), 1835–1848. https://doi.org/10.14778/ 3407790.3407793   
[47] Yi Zhang and Zachary G. Ives. 2020. Finding Related Tables in Data Lakes for Interactive Data Science. In SIGMOD 2020. Association for Computing Machinery, 1951–1966. https://doi.org/10.1145/3318464.3389726

[48] Ziqi Zhang. 2017. Effective and efficient Semantic Table Interpretation using TableMiner+. Semantic Web 8, 6 (2017), 921–957. https://doi.org/10.3233/SW-160242   
[49] Erkang Zhu, Fatemeh Nargesian, Ken Q. Pu, and Renée J. Miller. 2016. LSH Ensemble: Internet-Scale Domain Search. Proc. VLDB Endow. 9, 12 (Aug 2016), 1185–1196. https://doi.org/10.14778/2994509.2994534