# Exploratory Data Analysis (EDA) Findings

This report compiles data-backed statistical findings for all 20 analytical questions. Every number is computed strictly from the cleaned dataset (`Data/housing_clean.csv`).

### Q1: Price Distribution
- **Finding**: Mean Rs245.74L, median Rs240.02L, std Rs136.14L, range Rs10L-Rs500L. Broad and roughly flat across the range, not bell-shaped.

### Q2: Size Distribution
- **Finding**: 500-5000 sqft, mean 2,749.8, median 2,747.0. Essentially a perfect uniform spread.

### Q3: Price per SqFt by Property Type
- **Finding**: Mean price per sqft: Apartment Rs11,810, Independent House Rs11,857, Villa Rs11,776. Spread 0.7% -- no type premium.

### Q4: Size vs Price
- **Finding**: Pearson r = 0.097. Weak, positive and nearly negligible. Larger homes are not meaningfully pricier.

### Q5: Outliers
- **Finding**: 0 listings (0.00%) exceed the IQR cap of Rs32,764/sqft; 0 size outliers.

### Q6: Price per SqFt by State
- **Finding**: Rajasthan Rs11,727 to Karnataka Rs11,951. 1.9% spread.

### Q7: Average Price by City
- **Finding**: Rs242.22L (Cuttack) to Rs249.51L (Bangalore). Every city clusters near Rs245.81L.

### Q8: Median Age by Locality
- **Finding**: Median age 18-21 years in top 20 localities (overall median 19 years).

### Q9: BHK Mix by City
- **Finding**: BHK configuration is uniformly distributed: 1 BHK 20.1%, 2 BHK 19.9%, 3 BHK 20.0%, 4 BHK 19.9%, 5 BHK 20.1%

### Q10: Price Trend for Top Localities
- **Finding**: Top locality (Locality_207) averages Rs13,048/sqft. Cross-sectional pricing across build years shows no vintage appreciation.

### Q11: Correlation Heatmap
- **Finding**: Strongest pair: Price_in_Lakhs vs Future_Price_5Y r = 0.992 (by construction). Physical attributes show |r| < 0.05.

### Q12: Nearby Schools vs Price
- **Finding**: Rs11,841 (1 school) to Rs11,794 (10 schools). 1.8% spread.

### Q13: Nearby Hospitals vs Price
- **Finding**: Rs11,702-Rs11,964. 2.2% spread -- flat.

### Q14: Price by Furnishing Status
- **Finding**: Furnished Rs245.72L  Semi-furnished Rs245.22L  Unfurnished Rs246.26L. 0.4% spread.

### Q15: Price per SqFt by Facing
- **Finding**: East Rs11,797  North Rs11,793  South Rs11,790  West Rs11,878. 0.7% spread.

### Q16: Count by Owner Type
- **Finding**: Builder 83,253 (33.3%)  Owner 83,268 (33.3%)  Broker 83,479 (33.4%)

### Q17: Count by Availability Status
- **Finding**: Ready_to_Move 124,965 (50.0%)  Under_Construction 125,035 (50.0%)

### Q18: Parking vs Price
- **Finding**: With parking Rs245.93L vs without Rs245.55L. 0.2% spread.

### Q19: Amenities vs Price
- **Finding**: Rs11,803/sqft (1 amenity) to Rs11,854/sqft (all 5). 0.5% spread -- flat.

### Q20: Transport vs Price and Investment Label
- **Finding**: Price flat (High Rs11,785 vs Low Rs11,842, 0.5% spread). Good Investment rate: High 42.5% vs Low 7.3% -- circular: transport is an input to the label rule.

