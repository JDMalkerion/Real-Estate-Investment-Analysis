# Exploratory Data Analysis (EDA) Findings

This report compiles data-backed statistical findings for all 20 analytical questions.
Every number is computed strictly from the cleaned dataset (`Data/housing_clean.csv`).

### Q1: Price Distribution
- **Finding**: Property prices average ₹245.74 Lakhs (median ₹240.02 Lakhs, std ₹136.14 Lakhs) across a range of ₹10.00L to ₹500.00L.

### Q2: Size Distribution
- **Finding**: Property sizes span 500 to 5,000 SqFt with a mean of 2,749.8 SqFt and median of 2,747.0 SqFt, showing near-uniform spread.

### Q3: Price per SqFt by Property Type
- **Finding**: Mean price per sqft is nearly uniform across types: Apartment ₹11,810/sqft, Independent House ₹11,857/sqft, Villa ₹11,776/sqft (no strong signal of type-driven pricing premium).

### Q4: Size vs Price Relationship
- **Finding**: Size vs Price displays Pearson r = 0.0970 (no strong signal of linear dependence in this dataset, with prices uniformly distributed across unit sizes).

### Q5: Outliers in Price per SqFt and Size
- **Finding**: Price per SqFt exhibited right-skewed outliers with 19,723 records (7.89%) at the upper IQR threshold (₹32,764/sqft), whereas Size in SqFt had 0 outliers beyond IQR limits.

### Q6: Average Price per SqFt by State
- **Finding**: State-level average price per sqft ranges narrowly from ₹11,727 (Rajasthan) to ₹11,951 (Karnataka), showing no strong signal of statewide geographical disparity.

### Q7: Average Property Price by City
- **Finding**: Average property price across cities is tightly clustered around ₹245.81L, with Bangalore highest at ₹249.51L and Cuttack lowest at ₹242.22L.

### Q8: Median Property Age by Locality
- **Finding**: Across the top 20 most sampled localities, median property age ranges between 17 and 20 years (overall sample median 18 years).

### Q9: BHK Distribution Across Cities
- **Finding**: BHK configuration is uniformly distributed across major cities: 1 BHK (20.1%), 2 BHK (19.9%), 3 BHK (20.0%), 4 BHK (19.9%), and 5 BHK (20.1%).

### Q10: Price Trend for Top 5 Expensive Localities
- **Finding**: The top locality (Locality_207) averages ₹13,048/sqft; cross-sectional pricing across build years (1990-2023) shows stationary fluctuations without secular vintage appreciation.

### Q11: Correlation Heatmap of Numeric Features
- **Finding**: Strongest linear correlation is between Price_in_Lakhs and Future_Price_5Y (r = 0.991), while raw physical attributes show near-zero cross-correlations (|r| < 0.05).

### Q12: Nearby Schools vs Price per SqFt
- **Finding**: Pearson r = -0.0019; average price per sqft remains virtually flat from 1 school (₹11,841) to 10 schools (₹11,794) (no strong signal).

### Q13: Nearby Hospitals vs Price per SqFt
- **Finding**: Pearson r = -0.0004; average price per sqft is unaffected by hospital counts, ranging from ₹11,791 to ₹11,702 (no strong signal).

### Q14: Price by Furnished Status
- **Finding**: Average prices across furnishing categories are: Furnished ₹245.72L, Semi-furnished ₹245.22L, Unfurnished ₹246.26L (no strong signal of furnishing premium).

### Q15: Price per SqFt by Facing Direction
- **Finding**: Mean price per sqft across orientations shows negligible variance: North ₹11,793, South ₹11,790, East ₹11,797, West ₹11,878 (no strong signal).

### Q16: Inventory Share by Owner Type
- **Finding**: Listings are split almost equally across owner types: Broker 83,479 (33.4%), Owner 83,268 (33.3%), Builder 83,253 (33.3%).

### Q17: Inventory by Availability Status
- **Finding**: Availability is split evenly: Under Construction 125,035 (50.0%) vs Ready to Move 124,965 (50.0%).

### Q18: Parking Space vs Price
- **Finding**: Properties with designated parking average ₹245.93L vs ₹245.55L without parking (a difference of ₹0.38L; no strong signal).

### Q19: Amenities vs Price per SqFt
- **Finding**: Properties with 1 amenity average ₹11,803/sqft while properties with all 5 amenities average ₹11,854/sqft (flat across counts; no strong signal).

### Q20: Public Transport Accessibility vs Price and Investment Label
- **Finding**: Transport accessibility has negligible correlation with spot price (High: ₹11,785/sqft vs Low: ₹11,842/sqft), but profoundly shifts Good Investment likelihood: High access properties have 42.8% qualification vs 7.4% for Low access due to infrastructure scoring criteria.

