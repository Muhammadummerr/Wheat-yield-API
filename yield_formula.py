def estimate_yield(wheat_heads, avg_grain_weight_g=1.4, area_m2=1):
    return (avg_grain_weight_g * wheat_heads * 10000) / area_m2  # kg/ha
