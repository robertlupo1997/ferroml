---
name: netsuite-financial-mapping
description: Mapping between NetSuite chart of accounts and standard financial statement presentation. Use when querying NetSuite financial data, creating P&L statements, analyzing business line performance, or discussing revenue, costs, payroll, or any financial metrics that require knowing which NetSuite accounts to query or how to present financial data.
---

# NetSuite Financial Mapping

## Overview

This skill provides the mapping between your company's NetSuite chart of accounts and standard financial statement presentation. It enables Claude to correctly query NetSuite data and present it in the proper P&L format that matches your business expectations.

## When to Use This Skill

Use this skill for any query involving:
- Financial statement preparation (P&L, Income Statement)
- Revenue analysis by business line
- Cost of Sales (COS) analysis
- Payroll cost calculations
- Business line margin analysis
- Budget vs actual comparisons
- Any question about "where does [financial item] live in NetSuite?"

## Core Capabilities

### 1. Account Structure Knowledge

The skill provides complete knowledge of the chart of accounts structure:

**Business Lines:**
- Core (40000, 50xxx)
- Emerging (41000, 51xxx)
- Growth (42000, 52xxx)
- New Growth (42500, 525xx)
- Property Management (43000, 53xxx)
- Construction Management (44000)

**Account Categories:**
- Revenue accounts (4xxxx)
- Cost of Sales accounts (5xxxx)
- Operating Expense accounts (6xxxx)
- Other Income/Expense (7xxxx)
- Tax accounts (8xxxx)

### 2. Payroll Account Mapping

The skill understands how detailed payroll accounts roll up to consolidated categories:

- Detailed accounts split by: Business Line × Role (Producer/Support) × Type (Expense/Taxes/Benefits/Bonus)
- Roll-up accounts: 60300 (Payroll Expense), 60301 (Payroll Taxes), 60302 (Payroll Benefits), 60303 (Payroll Bonuses)

### 3. Financial Statement Presentation

The skill knows the standard P&L format for presenting NetSuite data:
- Revenue by business line
- Cost of Sales by business line with subcategories
- Operating Expenses by functional category
- Margin calculations (Gross, Operating, Net)

## Quick Reference

### Common Account Lookups

**Revenue by Business Line:**
- Core: 40000
- Emerging: 41000
- Growth: 42000
- New Growth: 42500
- Property Management: 43000
- Construction Management: 44000

**Total Payroll for a Business Line:**

For Core:
```
Producer Payroll = 50004 + 50006 + 50007 + 50005
Support Payroll = 50008 + 50010 + 50011 + 50009
Total Core Payroll = Producer + Support
```

Pattern applies to other lines (51xxx for Emerging, 52xxx for Growth, etc.)

### NetSuite Query Patterns

**Querying Revenue:**
```sql
SELECT SUM(amount)
FROM transaction
WHERE account IN (40000, 41000, 42000, 42500, 43000, 44000)
AND trandate BETWEEN [start] AND [end]
GROUP BY account
```

**Querying Cost of Sales by Business Line:**
```sql
SELECT account, SUM(amount)
FROM transaction
WHERE account >= 50000 AND account < 60000
AND trandate BETWEEN [start] AND [end]
GROUP BY account
```

**Querying Total Payroll:**
```sql
SELECT SUM(amount)
FROM transaction
WHERE account IN (
  -- List specific payroll accounts or use ranges
  50004, 50006, 50007, 50005, 50008, 50010, 50011, 50009,
  51004, 51006, 51007, 51005, 51008, 51010, 51011, 51009,
  -- etc.
)
AND trandate BETWEEN [start] AND [end]
```

## Workflow for Financial Queries

### Step 1: Understand the Request

Identify what financial information is being requested:
- Specific business line? (Core, Emerging, Growth, etc.)
- Type of data? (Revenue, COS, Payroll, etc.)
- Time period?
- Level of detail? (Summary vs detailed)

### Step 2: Load Reference Material

Based on the request, load the appropriate reference documents:

**For account lookups or general structure questions:**
```
Read: references/chart-of-accounts.md
```

**For payroll-related queries:**
```
Read: references/payroll-mappings.md
```

**For P&L statement creation:**
```
Read: references/pl-template.md
```

### Step 3: Query NetSuite

Use the NetSuite tools to query the appropriate accounts:

```
Use ns_runReport or ns_runCustomSuiteQL to pull data from identified accounts
```

### Step 4: Present Results

Format the data according to the standard P&L template:
- Group accounts by category
- Calculate subtotals and margins
- Present in the standard format from pl-template.md

## Example Queries

### Example 1: "What was our Core Revenue last quarter?"

**Analysis:**
- Business line: Core
- Metric: Revenue
- Account: 40000

**Query:**
```sql
SELECT SUM(amount)
FROM transaction
WHERE account = 40000
AND trandate BETWEEN '2024-07-01' AND '2024-09-30'
```

### Example 2: "Show me total payroll costs for the Growth business line"

**Analysis:**
- Business line: Growth
- Category: Payroll (all types)
- Accounts: 52004, 52006, 52007, 52005 (Producer) + 52008, 52010, 52011, 52009 (Support)

**Query:**
```sql
SELECT account, SUM(amount)
FROM transaction
WHERE account IN (52004, 52005, 52006, 52007, 52008, 52009, 52010, 52011)
AND trandate BETWEEN [start] AND [end]
GROUP BY account
```

**Presentation:**
```
Growth Business Line Payroll Costs

Producer Payroll:
  Payroll Expense (52004):    $XXX,XXX
  Payroll Taxes (52006):      $XX,XXX
  Payroll Benefits (52007):   $XX,XXX
  Payroll Bonus (52005):      $XX,XXX
  Subtotal Producer:          $XXX,XXX

Support Payroll:
  Payroll Expense (52008):    $XXX,XXX
  Payroll Taxes (52010):      $XX,XXX
  Payroll Benefits (52011):   $XX,XXX
  Payroll Bonus (52009):      $XX,XXX
  Subtotal Support:           $XXX,XXX

Total Growth Payroll:         $XXX,XXX
```

### Example 3: "Create a P&L for last month"

**Workflow:**
1. Load references/pl-template.md for structure
2. Load references/chart-of-accounts.md for account lists
3. Query NetSuite for all revenue accounts (40000-44000)
4. Query NetSuite for all COS accounts (50000-59999)
5. Query NetSuite for all operating expense accounts (60000-69999)
6. Query NetSuite for other income/expense (70000-79999) and taxes (80000+)
7. Format results according to the P&L template
8. Calculate margins and subtotals

## Important Notes

### Account Number Patterns

The account numbering follows a logical pattern:

**Cost of Sales (5xxxx):**
- First digit after 5 = Business line (0=Core, 1=Emerging, 2=Growth, 25=New Growth)
- Last 2-3 digits = Cost category

**Payroll within COS:**
- xx004 = Producer Expense
- xx005 = Producer Bonus
- xx006 = Producer Taxes
- xx007 = Producer Benefits
- xx008 = Support Expense
- xx009 = Support Bonus
- xx010 = Support Taxes
- xx011 = Support Benefits

### Data Quality Checks

When pulling NetSuite data, verify:
- Date ranges are correct (fiscal year may differ from calendar year)
- Account numbers match exactly (typos cause incorrect results)
- All relevant accounts are included in queries
- Subtotals match expected patterns

### Fiscal Year

**CRITICAL:** Always ask the user for the fiscal year start month before making date-based queries. Do not assume fiscal year = calendar year.

## Reference Files

This skill includes three detailed reference documents:

**references/chart-of-accounts.md**
Complete chart of accounts with all 150+ accounts organized by business line and category.

**references/payroll-mappings.md**
Detailed mapping of how individual payroll accounts roll up to consolidated payroll categories, with account number patterns and calculation formulas.

**references/pl-template.md**
Standard P&L statement format showing exactly how to present NetSuite data in consolidated income statement format, including formatting guidelines and presentation standards.
