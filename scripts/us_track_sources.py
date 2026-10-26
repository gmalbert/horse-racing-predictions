"""Verified track entry points and provider identifiers (checked 2026-10-25).

Track websites commonly delegate entries to Equibase or USTA. Query those
dated cards first; retain the official landing page as an independent fallback.
"""

def track(name, url, *, equibase=None, usta=None):
    return {"name": name, "url": url, "equibase": equibase, "usta": usta}


TB_TRACKS = {
    "TAM": track("Tampa Bay Downs", "https://www.tampabaydowns.com/racing/entries/", equibase="TAM"),
    "CT": track("Charles Town", "https://www.hollywoodcasinocharlestown.com/racing", equibase="CT"),
    "MNR": track("Mountaineer", "https://www.cnty.com/mountaineer/racing/", equibase="MNR"),
    "CBY": track("Canterbury Park", "https://www.canterburypark.com/racing/", equibase="CBY"),
    "PID": track("Presque Isle Downs", "https://www.presqueisledowns.com/race-information/", equibase="PID"),
    "PRX": track("Parx", "https://www.parxracing.com/", equibase="PRX"),
    "PEN": track("Penn National", "https://www.hollywoodpnrc.com/racing", equibase="PEN"),
    "ELP": track("Ellis Park", "https://www.ellisparkowensborogaming.com/racing/", equibase="ELP"),
    "LS": track("Lone Star Park", "https://www.lonestarpark.com/entries/", equibase="LS"),
    "HOU": track("Sam Houston", "https://www.shrp.com/racing", equibase="HOU"),
}
QH_TRACKS = {
    "LA": track("Los Alamitos", "https://www.losalamitos.com/ECE.aspx?ver=1", equibase="LA"),
    "RUD": track("Ruidoso Downs", "https://www.raceruidoso.com/", equibase="RUI"),
    "DLD": track("Delta Downs", "https://deltadowns.boydgaming.com/racing/racing-information", equibase="DED"),
    "EVD": track("Evangeline Downs", "https://evangelinedowns.boydgaming.com/racing", equibase="EVD"),
    "ZIA": track("Zia Park", "https://www.ziaparkcasino.com/racing", equibase="ZIA"),
    "SUN": track("Sunland Park", "https://sunland-park.com/racing/", equibase="SUN"),
}
H_TRACKS = {
    "NTH": track("Northfield Park", "https://mgmnorthfieldpark.mgmresorts.com/en/racing.html", usta="Nfld"),
    "SCD": track("Scioto Downs", "https://www.caesars.com/scioto-downs/racing", usta="ScD"),
    "RSC": track("Rosecroft", "https://www.rosecroft.com/", usta="RcR"),
    "RUN": track("Running Aces", "https://runaces.com/racing/live-racing/", usta="Aces"),
    "PLN": track("Plainridge Park", "https://www.plainridgeparkcasino.com/racing", usta="PRc"),
    "CAL": track("Cal Expo", "https://calxharness.com/", usta="CalX"),
}
EXTRA_TRACKS = {
    code: track(name, url, equibase=code) for code, name, url in [
        ("GP", "Gulfstream Park", "https://www.gulfstreampark.com/"),
        ("SA", "Santa Anita", "https://www.santaanita.com/"),
        ("DMR", "Del Mar", "https://www.dmtc.com/"),
        ("OP", "Oaklawn", "https://www.oaklawn.com/"),
        ("FG", "Fair Grounds", "https://www.fairgroundsracecourse.com/"),
        ("MTH", "Monmouth Park", "https://www.monmouthpark.com/"),
        ("CD", "Churchill Downs", "https://www.churchilldowns.com/"),
        ("KEE", "Keeneland", "https://www.keeneland.com/"),
    ]
}
EXTRA_TRACKS.update({
    "MDL": track("Meadowlands", "https://playmeadowlands.com/", usta="M"),
    "YON": track("Yonkers", "https://empirecitycasino.mgmresorts.com/en/racing.html", usta="YR"),
    "HOO": track("Hoosier Park", "https://www.caesars.com/harrahs-hoosier-park/racing", usta="HoP"),
})

ALL_TRACKS = {**TB_TRACKS, **QH_TRACKS, **H_TRACKS, **EXTRA_TRACKS}
