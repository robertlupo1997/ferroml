# Payroll Account Mappings

This document shows how detailed payroll accounts map to consolidated payroll categories for financial reporting.

## Overview

Payroll costs are tracked at a granular level by:
- **Business Line** (Core, Emerging, Growth, New Growth)
- **Role Type** (Producer vs Support)
- **Payroll Category** (Expense, Taxes, Benefits, Bonuses, Commissions)

These detailed accounts roll up to general payroll categories (60300-60303) for consolidated reporting.

## Payroll Mapping Structure

### Payroll Expense Accounts

Detailed accounts that roll up to **60300 - Payroll Expense**:

| Account | Description |
|---------|-------------|
| 50004 | Core COS Producer Payroll Expense |
| 50008 | Core COS Support Payroll Expense |
| 51004 | Emerging COS Producer Payroll Expense |
| 51008 | Emerging COS Support Payroll Expense |
| 52004 | Growth COS Producer Payroll Expense |
| 52008 | Growth COS Support Payroll Expense |
| 52504 | New Growth COS Producer Payroll Expense |
| 52508 | New Growth COS Support Payroll Expense |
| 53300 | Management COS & Accounting Payroll Expense |

**Roll-up Account:** 60300 - Payroll Expense

### Payroll Tax Accounts

Detailed accounts that roll up to **60301 - Payroll Taxes**:

| Account | Description |
|---------|-------------|
| 50006 | Core COS Producer Payroll Taxes |
| 50010 | Core COS Support Payroll Taxes |
| 51006 | Emerging COS Producer Payroll Taxes |
| 51010 | Emerging COS Support Payroll Taxes |
| 52006 | Growth COS Producer Payroll Taxes |
| 52010 | Growth COS Support Payroll Taxes |
| 52506 | New Growth COS Producer Payroll Taxes |
| 52510 | New Growth COS Support Payroll Taxes |
| 53310 | Management COS & Accounting Payroll Taxes |

**Roll-up Account:** 60301 - Payroll Taxes

### Payroll Benefits Accounts

Detailed accounts that roll up to **60302 - Payroll Benefits**:

| Account | Description |
|---------|-------------|
| 50007 | Core COS Producer Payroll Benefits |
| 50011 | Core COS Support Payroll Benefits |
| 51007 | Emerging COS Producer Payroll Benefits |
| 51011 | Emerging COS Support Payroll Benefits |
| 52007 | Growth COS Producer Payroll Benefits |
| 52011 | Growth COS Support Payroll Benefits |
| 52507 | New Growth COS Producer Payroll Benefits |
| 52511 | New Growth COS Support Payroll Benefits |
| 53320 | Management COS & Accounting Benefits |

**Roll-up Account:** 60302 - Payroll Benefits

### Payroll Bonus Accounts

Detailed accounts that roll up to **60303 - Payroll Bonuses**:

| Account | Description |
|---------|-------------|
| 50005 | Core COS Producer Payroll Bonus |
| 50009 | Core COS Support Payroll Bonus |
| 51005 | Emerging COS Producer Payroll Bonus |
| 51009 | Emerging COS Support Payroll Bonus |
| 52005 | Growth COS Producer Payroll Bonus |
| 52009 | Growth COS Support Payroll Bonus |
| 52505 | New Growth COS Producer Payroll Bonus |
| 52509 | New Growth COS Support Payroll Bonus |
| 53305 | Management COS & Accounting Bonus |

**Roll-up Account:** 60303 - Payroll Bonuses

### Commission Accounts

Commission accounts by business line:

| Account | Description |
|---------|-------------|
| 50003 | Core Commissions |
| 51003 | Emerging Commissions |
| 52003 | Growth Commissions |
| 52503 | New Growth Commissions |

**Note:** Commissions may also be tracked under general payroll categories in some contexts.

## Payroll Account Patterns

### By Business Line

**Core (50xxx):**
- Producer: 50004 (Expense), 50005 (Bonus), 50006 (Taxes), 50007 (Benefits)
- Support: 50008 (Expense), 50009 (Bonus), 50010 (Taxes), 50011 (Benefits)

**Emerging (51xxx):**
- Producer: 51004 (Expense), 51005 (Bonus), 51006 (Taxes), 51007 (Benefits)
- Support: 51008 (Expense), 51009 (Bonus), 51010 (Taxes), 51011 (Benefits)

**Growth (52xxx):**
- Producer: 52004 (Expense), 52005 (Bonus), 52006 (Taxes), 52007 (Benefits)
- Support: 52008 (Expense), 52009 (Bonus), 52010 (Taxes), 52011 (Benefits)

**New Growth (525xx):**
- Producer: 52504 (Expense), 52505 (Bonus), 52506 (Taxes), 52507 (Benefits)
- Support: 52508 (Expense), 52509 (Bonus), 52510 (Taxes), 52511 (Benefits)

### Account Number Pattern

For Cost of Sales payroll accounts:
- **[5][Business Line][00][Category]**
  - Business Line: 0=Core, 1=Emerging, 2=Growth, 25=New Growth
  - Category: 4/8=Expense, 5/9=Bonus, 6/10=Taxes, 7/11=Benefits
  - Even numbers (4,6,7) = Producer
  - Odd numbers (8,9,10,11) = Support

## Total Payroll Cost Calculation

To calculate total payroll costs for a business line:

**Total Payroll = Expense + Taxes + Benefits + Bonuses**

Example for Core business line:
```
Core Producer Payroll = 50004 + 50006 + 50007 + 50005
Core Support Payroll = 50008 + 50010 + 50011 + 50009
Total Core Payroll = Core Producer Payroll + Core Support Payroll
```

## Special Accounts

**13200 - Due from Ally Capital**: This account appears in some payroll mapping contexts but represents an intercompany receivable rather than a payroll expense.

**70300-70303 - New Growth Payroll (Operating Expense)**: These accounts track New Growth payroll in the operating expense section rather than Cost of Sales.

## Usage in Financial Reporting

When generating financial statements:

1. **Detailed P&L**: Show individual accounts by business line and role
2. **Summary P&L**: Roll up to categories (60300-60303)
3. **Business Line Analysis**: Sum all payroll accounts for a specific business line
4. **Margin Analysis**: Include all COS payroll when calculating gross margin by business line
