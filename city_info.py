"""City information and population data for e-scooter providers."""

# Normalize combined city names to a consistent format (keys are lowercase)
CITY_NORMALIZATION = {
    # Tübingen/Reutlingen area - standardize to "Reutlingen/Tübingen"
    "tübingen/reutlingen": "Reutlingen/Tübingen",
    "reutlingen/tübingen": "Reutlingen/Tübingen",
    "reutlingen_tübingen": "Reutlingen/Tübingen",
    "reutlingen_tubingen": "Reutlingen/Tübingen",
    "reutlingen_tuebingen": "Reutlingen/Tübingen",
    "tuebingen_reutlingen": "Reutlingen/Tübingen",
    "tübingen_reutlingen": "Reutlingen/Tübingen",
    "reutlingen": "Reutlingen/Tübingen",
    "tübingen": "Reutlingen/Tübingen",
    "tubingen": "Reutlingen/Tübingen",
    "tuebingen": "Reutlingen/Tübingen",
    # Stuttgart suburbs
    "böblingen": "Stuttgart",
    "boblingen": "Stuttgart",
    "ludwigsburg": "Stuttgart",
    "renningen_malmsheim": "Stuttgart",
    # Zürich suburbs
    "opfikon": "Zürich",
    "uster": "Zürich",
    # Lowercase city names -> proper case
    "stuttgart": "Stuttgart",
    "karlsruhe": "Karlsruhe",
    "mannheim": "Mannheim",
    "zurich": "Zürich",
    "zuerich": "Zürich",
    "zürich": "Zürich",
    "basel": "Basel",
    "bern": "Bern",
    "saarbrucken": "Saarbrücken",
    "saarbruecken": "Saarbrücken",
    "heilbronn": "Heilbronn",
    "ulm": "Ulm",
    "mainz": "Mainz",
    "kaiserslautern": "Kaiserslautern",
    "pforzheim": "Pforzheim",
    "heidelberg": "Heidelberg",
    "freiburg": "Freiburg",
    "konstanz": "Konstanz",
    "friedrichshafen": "Friedrichshafen",
    "villingen_schwenningen": "Villingen-Schwenningen",
    "villingen-schwenningen": "Villingen-Schwenningen",
    "st_gallen": "St. Gallen",
    "winterthur": "Winterthur",
    "lindau": "Lindau",
    "uberlingen": "Überlingen",
    "ueberlingen": "Überlingen",
    "bregenz": "Bregenz",
    "frankfurt": "Frankfurt",
    "goppingen": "Göppingen",
    "goeppingen": "Göppingen",
    "schwabisch_gmund": "Schwäbisch Gmünd",
    "schwaebisch_gmuend": "Schwäbisch Gmünd",
}

VOI_DE_CLUSTERS = {
    "cluster_0_(49.49,8.47)": "Mannheim",
    "cluster_1_(49.01,8.40)": "Karlsruhe",
    "cluster_2_(48.78,9.16)": "Stuttgart",
    "cluster_3_(48.89,8.70)": "Pforzheim",
    "cluster_4_(48.51,9.13)": "Reutlingen/Tübingen",
    "cluster_5_(50.15,8.66)": "Frankfurt",
}

VOI_CH_CLUSTERS = {
    "cluster_0_(47.42,8.60)": "Zürich",
    "cluster_1_(46.95,7.44)": "Bern",
    "cluster_2_(47.51,7.70)": "Basel",
}

# Annual PT ridership (passengers per year, 2024 data)
CITY_PT_RIDERSHIP = {
    "Zürich": 670_000_000,        # ZVV 2024 - https://geschaeftsbericht.zvv.ch/zahlen-und-fakten/
    "Stuttgart": 285_000_000,     # SSB + S-Bahn combined
    "Reutlingen/Tübingen": 23_000_000,  # TüBus 2024 - https://www.swtue.de/ (Tübingen only, RSV not published)
}

# Population data (2024/2025 estimates)
CITY_POPULATIONS = {
    "Stuttgart": 775_000,       # Dec 2025, incl. merged suburbs (Böblingen 51k, Ludwigsburg 99.5k, Renningen 18.5k)
    "Mannheim": 309_000,        # 2024, mannheim.de
    "Karlsruhe": 308_000,       # 2024, Zensus adjusted
    "Freiburg": 234_000,        # 2024, Zensus adjusted
    "Heidelberg": 155_000,      # 2024, heidelberg.de
    "Heilbronn": 130_000,       # 2024, +1.1% Zensus
    "Ulm": 127_000,             # 2024, Zensus adjusted
    "Pforzheim": 135_000,       # 2024, +4.7% Zensus
    "Reutlingen": 116_500,      # 2024, reutlingen.de
    "Tübingen": 92_800,         # 2024, tuebingen.de
    "Reutlingen/Tübingen": 209_300,  # Combined (Reutlingen + Tübingen)
    "Ludwigsburg": 99_500,      # 2024
    "Konstanz": 85_500,         # 2024
    "Villingen-Schwenningen": 86_000,  # 2024
    "Friedrichshafen": 62_000,  # 2024
    "Böblingen": 51_000,        # 2024
    "Saarbrücken": 182_000,     # 2024, saarbruecken.de
    "Mainz": 220_000,           # 2024
    "Kaiserslautern": 100_000,  # 2024
    "Lindau": 26_000,           # 2024
    "Überlingen": 24_000,       # 2024
    "Renningen/Malmsheim": 18_500,  # 2024
    "Frankfurt": 773_000,       # 2024
    "Zürich": 507_000,          # Dec 2024, incl. merged suburbs (Opfikon 21.5k, Uster 36k)
    "Basel": 178_000,           # Dec 2024, bs.ch (Stadt Basel-Stadt)
    "Bern": 135_000,            # 2024
    "Winterthur": 118_000,      # 2024
    "St. Gallen": 80_000,       # 2024
    "Opfikon": 21_500,          # 2024 (part of Zürich metro)
    "Uster": 36_000,            # 2024 (part of Zürich metro)
    "Bregenz": 30_000,          # 2024
    "Göppingen": 58_000,        # 2024
    "Schwäbisch Gmünd": 62_000, # 2024
}

PROVIDER_CITIES = {
    "bolt_basel": "Basel",
    "bolt_karlsruhe": "Karlsruhe",
    "bolt_reutlingen_tuebingen": "Reutlingen/Tübingen",
    "bolt_stuttgart": "Stuttgart",
    "bolt_zurich": "Zürich",
    "dott_basel": "Basel",
    "dott_boblingen": "Böblingen",
    "dott_bregenz": "Bregenz",
    "dott_friedrichshafen": "Friedrichshafen",
    "dott_heidelberg": "Heidelberg",
    "dott_heilbronn": "Heilbronn",
    "dott_kaiserslautern": "Kaiserslautern",
    "dott_karlsruhe": "Karlsruhe",
    "dott_lindau": "Lindau",
    "dott_ludwigsburg": "Ludwigsburg",
    "dott_mainz": "Mainz",
    "dott_mannheim": "Mannheim",
    "dott_reutlingen": "Reutlingen",
    "dott_saarbrucken": "Saarbrücken",
    "dott_st_gallen": "St. Gallen",
    "dott_stuttgart": "Stuttgart",
    "dott_tubingen": "Tübingen",
    "dott_uberlingen": "Überlingen",
    "dott_ulm": "Ulm",
    "dott_winterthur": "Winterthur",
    "dott_zurich": "Zürich",
    "hopp_konstanz": "Konstanz",
    "lime_opfikon": "Opfikon",
    "lime_stuttgart": "Stuttgart",
    "lime_zurich": "Zürich",
    "lime_basel": "Basel",
    "lime_uster": "Uster",
    "voi_de": None,
    "voi_ch": None,
    "yoio_freiburg": "Freiburg",
    "zeus_freiburg": "Freiburg",
    "zeus_heidelberg": "Heidelberg",
    "zeus_konstanz": "Konstanz",
    "zeus_ludwigsburg": "Ludwigsburg",
    "zeus_renningen_malmsheim": "Renningen/Malmsheim",
    "zeus_villingen_schwenningen": "Villingen-Schwenningen",
}


def normalize_city(city):
    """Normalize combined city names to consistent format (case-insensitive)."""
    if city is None:
        return None
    # Create lowercase lookup
    city_lower = city.lower()
    for key, value in CITY_NORMALIZATION.items():
        if key.lower() == city_lower:
            return value
    return city


def extract_city(provider: "pd.Series", clustered_provider: "pd.Series") -> "pd.Series":
    """
    Vectorized extraction and normalization of city names.
    """
    import pandas as pd
    import numpy as np

    # Choose source: clustered_provider for VOI, otherwise provider
    is_voi = provider.str.contains('voi', na=False)
    source = np.where(is_voi, clustered_provider, provider)
    source = pd.Series(source, index=provider.index)

    # Extract city part: everything after first underscore
    # e.g., "bolt_stuttgart" -> "stuttgart", "zeus_villingen_schwenningen" -> "villingen_schwenningen"
    city_raw = source.str.split('_', n=1).str[1].str.lower()

    # Map to normalized city names
    return city_raw.map(CITY_NORMALIZATION)


def sanitize_city_name(city):
    """
    Normalize city name and convert to safe filename format.
    """
    if city is None:
        return None

    # First normalize/merge
    city = normalize_city(city)

    # Convert to lowercase and sanitize
    name = city.lower()
    name = name.replace('ü', 'ue').replace('ö', 'oe').replace('ä', 'ae').replace('ß', 'ss')
    name = name.replace('/', '_').replace(' ', '_').replace('.', '')

    return name


def get_city_info():
    """Get full city info dictionary with populations."""
    city_info = {}

    for provider, city in PROVIDER_CITIES.items():
        if city is not None:
            city_info[provider] = {
                "city": city,
                "population": CITY_POPULATIONS.get(city, None),
            }

    for cluster_id, city in VOI_DE_CLUSTERS.items():
        provider_key = f"voi_de_{cluster_id}"
        city_info[provider_key] = {
            "city": city,
            "population": CITY_POPULATIONS.get(city, None),
        }

    for cluster_id, city in VOI_CH_CLUSTERS.items():
        provider_key = f"voi_ch_{cluster_id}"
        city_info[provider_key] = {
            "city": city,
            "population": CITY_POPULATIONS.get(city, None),
        }

    return city_info


if __name__ == "__main__":
    info = get_city_info()
    print("City Info Dictionary:")
    print("=" * 60)
    for provider, data in sorted(info.items()):
        pop_str = f"{data['population']:,}" if data['population'] else "N/A"
        print(f"{provider:40} -> {data['city']:25} (pop: {pop_str})")
