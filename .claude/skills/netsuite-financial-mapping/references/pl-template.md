# Standard P&L Statement Template

This document defines the standard Profit & Loss statement format for presenting NetSuite financial data.

## Consolidated Income Statement Structure

```
REVENUE
  Core Revenue (40000)
  Emerging Revenue (41000)
  Growth Revenue (42000)
  New Growth Revenue (42500)
  Property Management Revenue (43000)
  Construction Management Revenue (44000)
────────────────────────────────
Total Revenue

COST OF SALES
  Core Cost of Sales
    General Deal Costs (50000)
    Travel Deal Costs (50001)
    Meals & Entertainment Deal Costs (50002)
    Commissions (50003)
    Producer Payroll (50004, 50006, 50007, 50005)
    Support Payroll (50008, 50010, 50011, 50009)
    Other Direct Costs (50012-50017)
  
  Emerging Cost of Sales
    General Deal Costs (51000)
    Travel Deal Costs (51001)
    Meals & Entertainment Deal Costs (51002)
    Commissions (51003)
    Producer Payroll (51004, 51006, 51007, 51005)
    Support Payroll (51008, 51010, 51011, 51009)
    Other Direct Costs (51012-51017)
  
  Growth Cost of Sales
    General Deal Costs (52000)
    Travel Deal Costs (52001)
    Meals & Entertainment Deal Costs (52002)
    Commissions (52003)
    Producer Payroll (52004, 52006, 52007, 52005)
    Support Payroll (52008, 52010, 52011, 52009)
    Other Direct Costs (52012-52017)
  
  New Growth Cost of Sales
    General Deal Costs (52500)
    Meals & Entertainment Deal Costs (52502)
    Commissions (52503)
    Producer Payroll (52504, 52506, 52507, 52505)
    Support Payroll (52508, 52510, 52511, 52509)
    Other Direct Costs (52512-52517)
  
  Management Cost of Sales
    Pursuit/Referral Fees (53000)
    Payroll (53300, 53310, 53320, 53305)
    Other Costs (53400-53800)
────────────────────────────────
Total Cost of Sales

GROSS PROFIT
────────────────────────────────

OPERATING EXPENSES
  Payroll & Benefits
    General & Admin Payroll (60000, 60001, 60002)
    Sales & Marketing Payroll (60100, 60101, 60102)
    Technology Payroll (60200)
  
  Occupancy
    Rent (60800)
    Operating Lease Amortization (60801)
    Operating Lease Interest (60802)
  
  Technology & Software
    Dues & Subscriptions (60600)
    Software Subscriptions (60602)
    Computer Expense (61800)
    Computer Software (61802)
  
  Office & General
    General Office Expenses (61600)
    Office Furniture (61602)
    Office Supplies (61603)
    Telephone (61700)
  
  Travel & Entertainment
    General Meetings & Conferences (62000)
    Travel (62100)
    Meals & Entertainment (62200)
  
  Marketing & Business Development
    Pursuit/Referral Fees (62600)
    Marketing (62800)
    Advertising (62801)
    Public Relations (62802)
    Signage (62803)
  
  Taxes & Insurance
    Taxes, Dues & Licenses (63000)
    Real Estate Taxes (63001)
    Property Taxes (63002)
    Franchise Tax (63003)
    Unemployment Tax (63004)
    Insurance (63100)
  
  Professional Services
    Legal (64001)
    Accounting (64002)
    Data Processing & Payroll (64003)
    Consulting (64004)
    Recruiting (64005)
  
  Other Operating Expenses
    Bank Charges (64100)
    Merchant Processing Fees (64101)
    Penalties & Interest (64200)
    Depreciation (61900)
────────────────────────────────
Total Operating Expenses

OPERATING INCOME
────────────────────────────────

OTHER INCOME (EXPENSE)
  Other Income (70000)
  Gain on Disposal of Assets (70001)
  Interest Income (70100)
  Other Expense (70200)
  New Growth Payroll Expenses (70300-70303)
  Loss on Disposal of Assets (70400)
  Interest Expense (70500)
────────────────────────────────
Total Other Income (Expense)

INCOME BEFORE TAXES
────────────────────────────────

Income Tax Expense (80000)
  Federal Income Tax (80001)
  State Income Tax (80002)
────────────────────────────────

NET INCOME
════════════════════════════════
```

## Business Line Reporting

When presenting results by business line, use this structure:

### Revenue by Business Line

| Business Line | Account | Amount |
|---------------|---------|--------|
| Core | 40000 | $X,XXX |
| Emerging | 41000 | $X,XXX |
| Growth | 42000 | $X,XXX |
| New Growth | 42500 | $X,XXX |
| Property Management | 43000 | $X,XXX |
| Construction Management | 44000 | $X,XXX |
| **Total Revenue** | | **$X,XXX** |

### Gross Margin by Business Line

For each business line, calculate:
- **Revenue** = Business line revenue account
- **Direct Costs** = Sum of all COS accounts for that line
- **Gross Profit** = Revenue - Direct Costs
- **Gross Margin %** = (Gross Profit / Revenue) × 100

| Business Line | Revenue | Direct Costs | Gross Profit | Margin % |
|---------------|---------|--------------|--------------|----------|
| Core | $X,XXX | $X,XXX | $X,XXX | XX% |
| Emerging | $X,XXX | $X,XXX | $X,XXX | XX% |
| Growth | $X,XXX | $X,XXX | $X,XXX | XX% |
| New Growth | $X,XXX | $X,XXX | $X,XXX | XX% |

## Summary-Level Reporting

For executive summaries or board reports, use this condensed format:

```
Revenue                        $X,XXX,XXX
Cost of Sales                 ($X,XXX,XXX)
────────────────────────────────────────
Gross Profit                   $X,XXX,XXX
Gross Margin %                      XX%

Operating Expenses
  Payroll & Benefits          ($X,XXX,XXX)
  Occupancy                   ($XXX,XXX)
  Technology                  ($XXX,XXX)
  Marketing & BD              ($XXX,XXX)
  Other Operating             ($XXX,XXX)
────────────────────────────────────────
Total Operating Expenses      ($X,XXX,XXX)

Operating Income               $X,XXX,XXX
Operating Margin %                  XX%

Other Income (Expense)         ($XXX,XXX)
────────────────────────────────────────
Income Before Taxes            $X,XXX,XXX

Income Tax Expense             ($XXX,XXX)
────────────────────────────────────────
Net Income                     $X,XXX,XXX
Net Margin %                        XX%
════════════════════════════════════════
```

## Presentation Guidelines

### Formatting Standards
- Use consistent currency formatting: $X,XXX,XXX
- Expenses shown in parentheses: ($X,XXX)
- Percentages shown to one decimal place: XX.X%
- Subtotals underlined with single line: ────
- Totals double-underlined: ════

### Period Headers
Always include clear period identification:
- "For the Month Ended [Date]"
- "For the Three Months Ended [Date]"
- "For the Year Ended [Date]"

### Comparative Reporting
When showing multiple periods:

| Line Item | Current Period | Prior Period | Variance $ | Variance % |
|-----------|----------------|--------------|------------|------------|
| Revenue | $X,XXX | $X,XXX | $XXX | X% |

### Key Metrics
Always calculate and display:
- **Gross Margin %** = Gross Profit / Revenue
- **Operating Margin %** = Operating Income / Revenue
- **Net Margin %** = Net Income / Revenue
- **EBITDA** = Operating Income + Depreciation (61900)

## Account Grouping Rules

### Payroll Roll-Ups
When summarizing payroll:
1. Group all COS payroll by business line
2. Group all operating payroll by department (G&A, Sales, Technology)
3. Include all components: Expense + Taxes + Benefits + Bonuses

### Cost of Sales Categories
For detailed COS reporting, group into:
1. **Variable Deal Costs**: General Deal Costs, Travel, Meals, Commissions
2. **Direct Labor**: Producer and Support Payroll (all components)
3. **Other Direct Costs**: Technology, Subscriptions, Telephone, etc.

### Operating Expense Categories
Standard categories:
1. **Compensation**: All operating payroll accounts
2. **Occupancy**: Rent and lease-related
3. **Technology**: Software, subscriptions, computer expenses
4. **Professional Services**: Legal, accounting, consulting
5. **G&A**: Office expenses, travel, marketing, etc.
