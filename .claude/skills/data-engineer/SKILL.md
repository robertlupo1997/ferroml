---
name: data-engineer
description: Build scalable data pipelines, ETL/ELT processes, and data infrastructure. Use when: (1) designing data architectures or lakehouse patterns, (2) building Spark/Kafka/Flink/Beam pipelines, (3) optimizing Snowflake/BigQuery/Redshift queries, (4) implementing Airflow/Prefect/Dagster orchestration, (5) setting up data quality frameworks, (6) cost-optimizing data platforms.
---

# Data Engineer

## Workflow

1. **Assess** - Identify sources, volumes, velocity, SLAs, consumers
2. **Design** - Choose architecture pattern, storage layer, processing engine
3. **Implement** - Build pipelines with proper error handling and idempotency
4. **Quality** - Add validation, completeness checks, anomaly detection
5. **Monitor** - Set up metrics, alerts, lineage tracking
6. **Optimize** - Tune for cost and performance iteratively

## Architecture Selection

| Pattern | Use When |
|---------|----------|
| Medallion (bronze/silver/gold) | Multi-stage refinement, Databricks/Delta Lake |
| Lambda | Need both batch accuracy + real-time speed |
| Kappa | Stream-first, reprocessing via replay |
| Data Mesh | Domain-oriented, decentralized ownership |

## Pipeline Patterns

- **Idempotency**: Use merge/upsert, not append. Track watermarks.
- **Checkpointing**: Enable recovery without full reprocessing
- **Schema evolution**: Use formats that support it (Parquet, Avro, Iceberg)
- **Partitioning**: By date/region for pruning; avoid over-partitioning small data
- **File sizing**: Target 128MB-1GB files; compact small files

## Quality Framework

```
Completeness  → Row counts, null checks, required fields
Consistency   → Cross-source reconciliation, referential integrity
Accuracy      → Business rule validation, range checks
Timeliness    → Freshness SLAs, pipeline latency tracking
Uniqueness    → Duplicate detection, key constraints
```

## Cost Optimization

- **Storage tiering**: Hot → warm → cold based on access patterns
- **Compute**: Spot/preemptible for batch, reserved for steady-state
- **Compression**: Snappy for speed, Zstd for ratio
- **Partition pruning**: Filter pushdown to skip irrelevant data
- **Materialized views**: Pre-compute expensive aggregations

## Orchestration Best Practices

- Separate DAGs by domain/SLA criticality
- Use sensors sparingly (prefer event-driven triggers)
- Implement circuit breakers for external dependencies
- Tag tasks for cost attribution
- Keep task duration <1hr for debuggability
