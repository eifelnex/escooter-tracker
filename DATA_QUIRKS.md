================================================================================
DATASET QUIRKS & ANOMALIES - PRESENTATION BULLET POINTS
================================================================================

## 1. VEHICLE ID RESETS WITHOUT RIDES (System Glitches)

• Found 1,222 suspicious ID resets across all cities (last 7 days)
  - Scooter disappears and new ID appears at EXACT same location
  - Time gap: Average 2.9 minutes (often 0 minutes - simultaneous!)
  - Distance: Average 2.5 meters (essentially GPS noise)
  - Battery change: Average 0.17% (virtually identical)

• Impact varies by provider:
  - Dott: ~90% of suspicious cases (especially Saarbrucken, Heilbronn)
  - Bolt: ~10% of suspicious cases
  - VOI: Minimal issues

• In Tubingen specifically:
  - 280 suspicious resets out of 5,307 disappearances (5.3%)
  - 74.6% happen within 10 meters
  - 66.1% happen within 5 minutes
  - Most occur during daytime hours (11 AM - 2 PM peak)

• Conclusion: These are API bugs or system glitches, NOT real rides


## 2. NIGHTLY BULK OPERATIONS (Scheduled Maintenance)

• Dott Saarbrucken: Massive nightly rebalancing operation
  - EXACTLY 1,700 scooters disappear at 01:34 AM every night
  - Occurs 6 consecutive nights with <1 second precision
  - Same scooters reappear at 03:34-03:35 AM with new IDs
  - Duration: ~2 hours (pickup -> recharge/rebalance -> return)

• Pattern detected:
  - 01:34:06 AM: Mass disappearance
  - 03:34-03:35 AM: Mass reappearance (first_seen)
  - Accounts for ~1,700 false "rides" per night
  - Represents entire Saarbrucken fleet being serviced

• Other bulk operations found:
  - Startup anomalies (Oct 28): 13,939 events at 11:49:32 AM
  - Nov 1 system resync: 1,674 events at 09:21 AM
  - Total bulk operations: 35,001 events (11.8% of all disappearances)


## 3. BATTERY ANALYSIS: RIDES vs MAINTENANCE

• Battery at disappearance (ride start):
  - Average: 58.8%
  - High battery (>50%): 58.2% of cases
  - Medium battery (20-50%): 36.5%
  - Low battery (<20%): 5.3%

• Battery at first_seen (ride end):
  - Average: 56.1%
  - Average drop: 2.7% per "ride"

• Analysis:
  - If maintenance: Would expect LOW battery at pickup (<30%)
  - Reality: 58% start with HIGH battery (>50%)
  - Conclusion: MOST disappearances are real rides, not charging pickups
  - Only ~5% low-battery cases are likely maintenance pickups

• Provider differences:
  - Bolt: 4.8% battery drop per ride (clear consumption)
  - Dott: 0.8% battery drop (masked by aggressive recharging)
  - Dott recharges 40.4 times per 100 rides (vs Bolt's 8.7 times)


## 4. DOTT'S BATTERY MANAGEMENT QUIRK

• Dott shows minimal battery drop (0.8%) but has LOWER range scooters:
  - Dott range: 25-27 km
  - Bolt range: 29-32 km
  - Dott consumes 3.98% per km (MORE than Bolt's 3.45%)

• Explanation: Aggressive post-ride "recharging"
  - 41% of Dott first_seen events followed by "recharge"
  - BUT: These are 3-6% battery adjustments, NOT full swaps
  - Average increase: Only 5.1% (not 80%+ like real battery swaps)
  - Likely battery voltage recovery or BMS recalibration

• Theory: Dott's Battery Management System (BMS)
  - Underestimates battery while scooter sits idle
  - Active riding provides better voltage readings
  - System recalibrates upward after ride (3-6% adjustment)
  - Masks true consumption pattern


## 5. RELOCATIONS WHILE "AVAILABLE" (Live Scooters Moving)

• Detected via location changes without disappearance
  - Change type: "transport" or "relocated_charged"
  - Scooters marked as "available" but moving >100 meters

• Use cases identified:
  - Rebalancing: Moving scooters between areas
  - Charging: Transport to charging facility while battery swapping
  - Maintenance: Moving to repair locations
  - Manual repositioning in high-demand areas

• Volume: Significant operational activity beyond just rides


## 6. VEHICLE ID ROTATION AFTER RIDES (GBFS v2.0 Privacy)

• Confirmed 94% ID rotation rate after disappearances
  - 997 disappearances analyzed
  - 0 reappeared with same vehicle ID (except 3 glitches)
  - Average ID lifespan: 14.3 hours (median: 4.3 hours)

• Distribution of ID lifespans:
  - 28.6% live <1 hour
  - 53.4% live 1-24 hours (most common)
  - Only 0.4% survive full tracking period (inactive scooters)

• Recharges do NOT trigger ID rotation:
  - 100% of vehicles kept same ID after recharge
  - Only rides (disappearances) cause new IDs

• Implication: Cannot track individual physical scooters long-term


## 7. DATA QUALITY SUMMARY

Total disappearances (8.22 days): 296,673

Breakdown:
• Bulk operations: 35,001 (11.8%) - NOT rides
• Suspicious ID resets: ~10,000 (3.4%) - System glitches
• Clean rides: ~251,672 (84.8%) - Legitimate usage

Estimated daily ride count:
• Raw average: 36,085 disappearances/day
• After filtering bulk: 32,543 rides/day
• After filtering ID resets: ~31,800 rides/day
• Confidence: 84.8% of disappearances are real rides


## 8. TEMPORAL PATTERNS

• Peak activity hours: 15:00-18:00 (3-6 PM)
  - 16:00 (4 PM) highest: ~2,700 rides/hour
  - Afternoon rush + leisure riding

• Low activity: 02:00-05:00 AM
  - ~3,000-4,400 rides total (across all cities)
  - Mostly maintenance operations at 01:34 AM

• Weekly pattern:
  - Friday peak: ~51,800 disappearances
  - Sunday lowest: ~25,600 disappearances
  - Weekend effect: ~50% drop on Sunday


## 9. LIMITATIONS & CAVEATS

• Cannot distinguish individual maintenance from rides
  - Single scooter pickup looks identical to rental
  - Only bulk operations (>100 at once) are detectable

• Vehicle IDs rotate (privacy feature)
  - Cannot track same physical scooter across rides
  - Cannot calculate per-scooter utilization accurately

• GBFS v3 doesn't solve these issues
  - Still no trip start/end data
  - Still no ride vs maintenance distinction
  - Would need MDS (Mobility Data Specification) for trip-level data

• Battery data has provider-specific quirks
  - Dott: BMS recalibration masks consumption
  - Different reporting accuracy between providers


## 10. RECOMMENDED REPORTING

Conservative estimate: "30,000-33,000 scooter usage events per day"
• Lower bound: 30,000 (assuming 10% maintenance)
• Best estimate: 31,800 (after filtering known anomalies)
• Upper bound: 33,000 (optimistic scenario)

Caveat: "Includes rides and some individual maintenance pickups that
cannot be distinguished from rides using GBFS data alone"

High-battery start rate (58%) strongly suggests majority are real rides.


================================================================================
TUBINGEN-SPECIFIC FINDINGS
================================================================================

• Total disappearances (7 days): 5,307
• Suspicious ID resets: 280 (5.3%)
• Clean rides estimate: 5,027
• Daily average: ~718 rides/day

• Provider breakdown:
  - Bolt: ~129/day
  - VOI: Not in Tubingen data shown
  - Dott: ~104/day (with BMS quirks)

• Data quality: Better than overall dataset (5.3% vs 12% suspicious rate)
• Peak hour: 17:00 (5 PM) with ~50 rides/hour
• Weekend dip: Sunday has ~50% lower activity than Friday
