+++
title = "A Learned Federated Query Compiler"
[extra]
latex = true
[[extra.authors]]
name = "Victor Giannakouris"
link = "https://www.linkedin.com/in/vgian/"
+++

# Intro
The goal of this project was to develop an optimizer for a _federated query_ compiler. A federated query, is an SQL 
query that consists of tables that span across different, individual database systems. One of the main drawbacks with 
the already existing federated query optimization approaches is the complexity of integration with new systems. 
The main reason is that the existing solutions need to obtain specific knowledge from the  externally connected 
databases by writing complex drivers (wrappers).

We propose a machine learning based solution that enables the easy integration of any SQL-based system, with _less than 
10 LoC_.

## Federated Query Compilation Challenges
A lot of interesting challenges arise in federated query compilation, due to the heterogeneous nature of the 
underlying infrastructure. One of the main challenges is the following: 
1. How could the query plan be split to further subqueries 
2. How the execution of these subqueries could be orchestrated across the different data sources in order utilize the external data systems and take advantage of their powerful implementations 
   (e.g., access methods and join algorithms)? 

In this project, we used Spark SQL as our baseline. We developed a federated

# What did you do?
We mainly focused on the optimizer of our federated query compiler. We implemented a federated query compiler which is
able to do optimize federated queries by splitting the execution of the initial query into individual, smaller parts
that are pushed down to the external database systems, achieving significant performance improvements. The optimizer
of our query compiler is based on a machine learning approach which treats the external systems as black-boxes,
profiles them and learns how to estimate the query execution cost of each individual system. Thus, our federated query
compiler is able to easily connect with any system that supports SQL, without needing any expertise of the underlying 
infrastructure.

## The Self-Learned Federated Query Compiler
We developed a query compiler that optimizes input queries with respect to the underlying table locations. To explain 
how our optimizer works, let's take a look at the following example. Consider the query:

```sql
SELECT AVG(A.salary)
FROM A, B, C
WHERE A.id = B.id
AND A.id = C.id
AND B.id = C.id
```
This query consists of a join between three tables: A, B and C. Let's assume three different data sources, namely DS1 
and DS2 and assume the following table locations:

- A -> DS1
- B -> DS2
- C -> DS2

In relational algebra, this join would look like the following: 
`A ⋈ B ⋈ C`. By default, a federation engine (like Spark SQL) would first transfer all tables A, B and C to its system
and then, it would execute the query its execution engine. However, more complex strategies could be followed for a query 
like this. For example, `A ⋈ B ⋈ C` could be split into two complement subqueries, `A` (scan) and `B ⋈ C` (join), where 
the first query which is just a scan would be executed as a simple scan from DS1. However, the second query (`B ⋈ C`) 
could be pushed down for local execution in `DS2`. But how do we know if pushing-down `B ⋈ C` to `DS2` would result to
a better performance compared to just fetching tables `B` and `C` to the federation engine (Spark SQL) and executing the 
join there?

## Learned Query Compilation
To answer the previous question, we use a machine learning model as our cost model that helps us evaluate the cost of 
a query without executing it. Using this model, we can easily evaluate whether it is better to push-down a query or not.
In order to do that, we use a unified vector space model that is able to transform SQL queries into vectors, and thus,
we can feed these queries to our model in order to train and predict the execution time of future queries. Thus, our
optimizer is able to leverage this machine learning model in order to decide which parts of the query should push-down 
or not.

## Query Compiler Design
Let's take a look at how our federated query compiler works. Similarly to any other compiler, we have a syntax parser
which takes as input an SQL query in raw text and produces and intermediate representation. Then, we perform further 
optimizations to this intermediate representation and then we pass the output optimized plan to the executor in order to 
evaluate the query and get the resulting rows.

### Compilation Lifecycle
The steps of compilation of a query consists of the following steps.
1. Parsing: We use Spark SQL query parser
2. Intermediate Representation: The first intermediate representation is a tree-like structure, where each parent node represents the join between two tables
3. Graph Representation: We tranform the aforementioned intermediate representation to a more modular graph structure, which gives us more freedom on create efficient plans.
4. Optimization: By examining the graph, we produce a the optimal (according to the cost model) tree-like query execution plan
5. Execution: We pass the final plan back to the federation engine for evaluation

We explain in more detail the intermediate representation that our compiler uses.

### Graph Intermediate Representation
Let's take again a look at the previous example
```sql
SELECT AVG(A.salary)
FROM A, B, C
WHERE A.id = B.id
AND A.id = C.id
AND B.id = C.id
```

If we think of this query as a graph, it would look as follows:
```
    A - B
    \   / 
      C
```

This is a fully-connected graph, as there are connections from all nodes to all nodes. As the final plan will be a binary tree,
that means that the final plan could be any of the following three trees:
```
1.  ⋈         2.  ⋈          3.  ⋈ 
   /  \          /  \           /  \
  A   ⋈         B   ⋈          C   ⋈ 
     / \           /  \           /  \
    B   C         A    C         A    B
```

However, we are mostly interested in plans that look like 1. The reason is that we are mainly interested in joins that 
can co-locate tables thus, so execution push-down is possible. Thus, we created a graph structure in which each node
represents a table, and each edge represents a join between two tables. We enrich more our graph representation by adding
information about the location of each table in the node, that is, in which data source the node resides. This allows
us to develop an algorithm can can parse this graph and generate possible trees that join tables that reside in the same
engine, when that's possible given the input query semantics (joins).

### Final Plan Generation
After processing the graph, we get a tree that collocates tables residing in the same engine by producing a tree-plan
that joins together the collocated tables. Then, we use our machine learning model in order to predict the execution cost
of joining the collocated tables in the machine that they reside on in the federation engine. We then produce the final
query execution plan which also contains information about where each node of the tree will be executed.

# Challenges
The trickiest part of the project, which is still not completely resolved is the cost estimation. The average accuracy of 
our model in predicting the cost of a given query is around 60%. However, in most of the cases in which we mispredict
the cost, the final plan that splits end distribues the query execution across the different engines is still better
than just using the federation engine for the query execution. We present some results that demonstrate this in the 
final section.

# Evaluation
In order to evaluate our system we use two benchmarks, [TPC-H](https://www.tpc.org/tpch/) and 
[Join-Order-Benchmark](https://github.com/gregrahn/join-order-benchmark). The experimental setup consisted of
one federation engine (Spark SQL) and two external systems, MySQL and PostgreSQL. 

## Cost Model Accuracy
We first try to evaluate the accuracy
of our machine learning cost model. For this experiment, we evaluate how well the cost model does in selecting the
appropriate execution engine (Spark SQL, Postgres or MySQL) in order to execute some part of the query. As mentioned in 
the previous section, the cost model accuracy is around 60%. However, in this experiment our model mispredicts in cases
where the performance difference is very low. By running the TPC-H benchmark, we find that the average performance of
running the queries in Spark without further optimization is 1150.57 ms. With further optimization and if we "manually"
select the optimal execution engine each time, the optimal performance will be 122.666. Using the engine selections
with our predictive model, the average query execution time is 152.7142, which is around the 1/7 of the query execution
time in Spark. We summarize these results in the following array:

| System | Average Execution Time |
| --- | --- |
| Spark SQL | 1150.57 ms |
| Optimal | 122.666 |
| Predicted | 152.7142 |

This experiment was the first simple proof-of-concept case in which all the tables reside in all engines,
that is, both MySQL and SparkSQL. We run this experiment in order to demonstrate the accuracy of our cost model
and to showcase that even in the case of multiple mispredictions, the final performance is close enough to the optimal.

## Performance
In order to measure the performance, we used the Join-Order-Benchmark. For those queries, we trained by changing the
number of the maximum tables (`table_limit`) that can be included in a subquery that is pushed down for local execution to an external 
engine. For example, assuming that tables `A`, `B` and `C` in the query `A ⋈ B ⋈ C` reside in the same engine, a 
`table_limit` 2 would enforce our optimizer to push-down to the external engine subquery like  `B ⋈ C` and a scan of `A`.
Then, the join between `A` and the result of `B ⋈ C` could be executed in the federation engine. The reason that we need
to adjust this number, is that if we push-down a large join that produces a relatively large intermediate result,
fetching this result from the external engine to the federation engine might result into big overheads that will make
the query execution slow. Using machine learning and also by verifying it empirically, we conclude for this benchmark that
a `table_limit` of three works the best for most of the queries. Sticking to that parameter we can achieve run most of the
queries to the ~ 60% of time, compared to the vanilla implementation of Spark SQL. However, our optimizer is still on a 
very immature stage. We believe that our ongoing extension in which the `table_limit`, as well as the selection
of the actual tables that will be included to a join will improve even better improvements.

## Conclusions and Future Steps
We presented a compiler that accelerates the execution of federated queries. We presented an intermediate
representation that follows a graph structure, includes location information about each table and we showed how 
we can leverage this structure in order to produce query execution plans that leverage the individual systems of the underlying
infrastructure. We also presented a learned cost model that assists the query planner, despite its average prediction
accuracy, it is able to achieve performance improvements close to the optimal plan. However, we still use a rule-based
optimizer that is assisted by our learned cost model, but there is still work to be done in order to create a robust 
federated query planner. Our future steps include the following:
- Include more complex features to the vectorized query representation in order to improve the cost model accuracy
- Develop a more sophisticated, dynamic programming based query planner that uses the learned cost model in order to provide better query execution plans
- Develop and evaluate self-learned planner using Reinforcement Learning.
