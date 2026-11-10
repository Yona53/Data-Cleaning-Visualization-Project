# Key Findings

## Cleaning summary
- Rows: 1,575 -> 1,500 (75 exact duplicates removed)
- Missing cells (excluding rating): 361 -> 0
- Price outliers (IQR x3 within category): 13 treated as bad entries and imputed with category median
- Invalid values (negative quantity, age > 100) set to missing, then imputed
- Ratings left missing on purpose: imputing would bias satisfaction scores
- Text standardized (region, category), "$" stripped from prices, mixed date formats parsed

## Insights
1. **Electronics** generates 55% of revenue, so inventory and marketing there matter most.
2. **West** is the top region by revenue.
3. Nov-Dec account for 16% of revenue (about 17% would be an even share), so there is no strong holiday peak; demand is fairly steady across the year.
4. Discount level shows little relationship with ratings, so deeper discounts do not clearly buy satisfaction.

## Recommendations
- Concentrate promotions and inventory on the top category.
- Test discount levels before widening them, since the data shows no rating benefit.
- Add input validation at the source (date format, price type, age range) to stop recurring data issues.
