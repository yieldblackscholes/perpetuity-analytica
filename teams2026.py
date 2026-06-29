"""2026 FIFA World Cup — the 48 qualified teams, their groups, and the
official final-draw seeding. `csv_name` is how a team appears in the
historical results dataset (martj42/international_results); `name` is the
display name used throughout the app.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class WCTeam:
    csv_name: str
    name: str
    code: str
    group: str
    flag: str


WC2026_TEAMS: list[WCTeam] = [
    # Group A
    WCTeam("Mexico", "Mexico", "MEX", "A", "🇲🇽"),
    WCTeam("South Africa", "South Africa", "RSA", "A", "🇿🇦"),
    WCTeam("South Korea", "South Korea", "KOR", "A", "🇰🇷"),
    WCTeam("Czech Republic", "Czech Republic", "CZE", "A", "🇨🇿"),
    # Group B
    WCTeam("Canada", "Canada", "CAN", "B", "🇨🇦"),
    WCTeam("Bosnia and Herzegovina", "Bosnia & Herz.", "BIH", "B", "🇧🇦"),
    WCTeam("Qatar", "Qatar", "QAT", "B", "🇶🇦"),
    WCTeam("Switzerland", "Switzerland", "SUI", "B", "🇨🇭"),
    # Group C
    WCTeam("Brazil", "Brazil", "BRA", "C", "🇧🇷"),
    WCTeam("Morocco", "Morocco", "MAR", "C", "🇲🇦"),
    WCTeam("Haiti", "Haiti", "HAI", "C", "🇭🇹"),
    WCTeam("Scotland", "Scotland", "SCO", "C", "🏴"),
    # Group D
    WCTeam("United States", "USA", "USA", "D", "🇺🇸"),
    WCTeam("Paraguay", "Paraguay", "PAR", "D", "🇵🇾"),
    WCTeam("Australia", "Australia", "AUS", "D", "🇦🇺"),
    WCTeam("Turkey", "Turkey", "TUR", "D", "🇹🇷"),
    # Group E
    WCTeam("Germany", "Germany", "GER", "E", "🇩🇪"),
    WCTeam("Curaçao", "Curaçao", "CUW", "E", "🇨🇼"),
    WCTeam("Ivory Coast", "Ivory Coast", "CIV", "E", "🇨🇮"),
    WCTeam("Ecuador", "Ecuador", "ECU", "E", "🇪🇨"),
    # Group F
    WCTeam("Netherlands", "Netherlands", "NED", "F", "🇳🇱"),
    WCTeam("Japan", "Japan", "JPN", "F", "🇯🇵"),
    WCTeam("Sweden", "Sweden", "SWE", "F", "🇸🇪"),
    WCTeam("Tunisia", "Tunisia", "TUN", "F", "🇹🇳"),
    # Group G
    WCTeam("Belgium", "Belgium", "BEL", "G", "🇧🇪"),
    WCTeam("Egypt", "Egypt", "EGY", "G", "🇪🇬"),
    WCTeam("Iran", "Iran", "IRN", "G", "🇮🇷"),
    WCTeam("New Zealand", "New Zealand", "NZL", "G", "🇳🇿"),
    # Group H
    WCTeam("Spain", "Spain", "ESP", "H", "🇪🇸"),
    WCTeam("Cape Verde", "Cape Verde", "CPV", "H", "🇨🇻"),
    WCTeam("Saudi Arabia", "Saudi Arabia", "KSA", "H", "🇸🇦"),
    WCTeam("Uruguay", "Uruguay", "URU", "H", "🇺🇾"),
    # Group I
    WCTeam("France", "France", "FRA", "I", "🇫🇷"),
    WCTeam("Senegal", "Senegal", "SEN", "I", "🇸🇳"),
    WCTeam("Iraq", "Iraq", "IRQ", "I", "🇮🇶"),
    WCTeam("Norway", "Norway", "NOR", "I", "🇳🇴"),
    # Group J
    WCTeam("Argentina", "Argentina", "ARG", "J", "🇦🇷"),
    WCTeam("Algeria", "Algeria", "ALG", "J", "🇩🇿"),
    WCTeam("Austria", "Austria", "AUT", "J", "🇦🇹"),
    WCTeam("Jordan", "Jordan", "JOR", "J", "🇯🇴"),
    # Group K
    WCTeam("Portugal", "Portugal", "POR", "K", "🇵🇹"),
    WCTeam("DR Congo", "DR Congo", "COD", "K", "🇨🇩"),
    WCTeam("Uzbekistan", "Uzbekistan", "UZB", "K", "🇺🇿"),
    WCTeam("Colombia", "Colombia", "COL", "K", "🇨🇴"),
    # Group L
    WCTeam("England", "England", "ENG", "L", "🏴"),
    WCTeam("Croatia", "Croatia", "CRO", "L", "🇭🇷"),
    WCTeam("Ghana", "Ghana", "GHA", "L", "🇬🇭"),
    WCTeam("Panama", "Panama", "PAN", "L", "🇵🇦"),
]

GROUPS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]

TEAM_BY_NAME = {t.name: t for t in WC2026_TEAMS}
TEAM_BY_CSV_NAME = {t.csv_name: t for t in WC2026_TEAMS}


def find_team(query: str) -> WCTeam | None:
    """Match a team by display name or code, case-insensitively, with a
    fallback to partial/prefix matching."""
    q = query.strip().lower()
    for t in WC2026_TEAMS:
        if t.name.lower() == q or t.code.lower() == q:
            return t
    for t in WC2026_TEAMS:
        if q and (q in t.name.lower() or t.name.lower().startswith(q)):
            return t
    return None


# Fixed Round-of-32 bracket template, faithful to the real World Cup draw
# structure: group winners never meet each other in the Round of 32, and a
# group's winner/runner-up sit in opposite halves of the bracket (so they can
# only meet again in the final). Codes:
#   ("W", group) -> group winner   ("R", group) -> runner-up
#   ("T", i)     -> the i-th best third-place team (0 = best)
R32_BRACKET: list[tuple[tuple, tuple]] = [
    (("W", "A"), ("T", 0)),
    (("R", "C"), ("R", "D")),
    (("W", "E"), ("T", 1)),
    (("W", "G"), ("R", "H")),
    (("W", "B"), ("T", 2)),
    (("R", "F"), ("R", "L")),
    (("W", "I"), ("T", 3)),
    (("W", "K"), ("R", "J")),
    (("W", "C"), ("T", 4)),
    (("R", "A"), ("R", "B")),
    (("W", "F"), ("T", 5)),
    (("W", "H"), ("R", "G")),
    (("W", "D"), ("T", 6)),
    (("R", "E"), ("R", "I")),
    (("W", "J"), ("T", 7)),
    (("W", "L"), ("R", "K")),
]

# Which group's winner sits on the other side of each ("T", i) slot, used to
# avoid (where possible) a third-place team facing its own group's winner.
TSLOT_WINNER_GROUP = {0: "A", 1: "E", 2: "B", 3: "I", 4: "C", 5: "F", 6: "D", 7: "L"}
