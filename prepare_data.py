import os
import nltk
from langchain.text_splitter import RecursiveCharacterTextSplitter

def ensure_nltk():
    # Download punkt only if missing to avoid repeated downloads
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)

def load_scraped(data_dir: str = "./fatrag_data") -> list[str]:
    scraped = []
    if os.path.isdir(data_dir):
        for filename in os.listdir(data_dir):
            if filename.endswith(".txt"):
                with open(os.path.join(data_dir, filename), "r", encoding="utf-8") as f:
                    scraped.append(f.read())
    return scraped

def build_chunks() -> list[str]:
    ensure_nltk()

    # Fatrag-aanpak-succesfactoren (nieuwe toevoeging, correct opgenomen)
    fatrag_succesfactoren = """
Aanpak DE NEXT STEP SINCERELY FATRAG
Ambitie Advies Activatie Aansluiten Afmaken
DE NEXT STEP STEP 1 STEP 2 STEP 3 STEP 4 STEP 5
STEP 1 Ambitie VOORDOEN SAMEN DOEN ZELF DOEN
Probleem vs. ambitie
Doel: Gewenst gedrag
Commitment 50% - 50% verantwoordelijkheid
Wil je dit echt? We zijn al begonnen
METEN VERSLAGLEGGING
AANGESLOTEN < 5%
Zelf niet willen veranderen
STEP 2 Advies VOORDOEN SAMEN DOEN ZELF DOEN
Uitvoer scan
Plan van aanpak
Delen & bespreken scanverslag
Objectieve, eerlijke spiegel van de situatie
Een realistisch advies
Het delen van het verslag kan reuring opleveren
METEN VERSLAGLEGGING
AANGESLOTEN 5-10%
Er kan weerstand komen op de aankondiging en onze aanwezigheid
STEP 3 Activatie VOORDOEN SAMEN DOEN ZELF DOEN
Kerngroep oprichten
Stuurgroep oprichten
Managementteam sessies
Eerste interventies vinden plaats
Klein beginnen en vanuit positiviteit op alle lagen
Focus ligt op ieders eigen gedrag
Andere aanpak dan men gewend is
METEN VERSLAGLEGGING
AANGESLOTEN 10-20%
STEP 4 Aansluiten VOORDOEN SAMEN DOEN ZELF DOEN
Kerngroepplan + meeting kerngroep
Vervolggroepen
Interventiekeuzes
Oprichten werkgroepen
Train de trainer
We gaan door, ook als het moeilijk wordt
Geen standaard programma's, doen wat nodig is
METEN VERSLAGLEGGING
AANGESLOTEN 20-80%
'Bescherm' het kernteam • Wisseling in MT • Interne politiek
STEP 5 Afmaken VOORDOEN SAMEN DOEN ZELF DOEN
Integratie in organisatie/afdeling processen
Managementcyclus | Coachcyclus | HR | Eigen trainingen
Borgingplan
Audit na x periode
We laten zo snel mogelijk los als het kan
AANGESLOTEN > 80%
Te snel denken dat we er al zijn
'Kies hier een vaste quote' Let's do this! SINCERELY FATRAG
"""

    # Provided documents (volledige bronnen voor RAG)
    provided_docs = [
        fatrag_succesfactoren,

        # Vacature Trainer Communicatie & Gedrag (volledige tekst)
        """
Vacature
Vacature: Trainer Communicatie & Gedrag
Bij Fatrag zijn wij op zoek naar een trainer communicatie en gedrag die onze Next Step aanpak tot leven kan brengen. Zie jij een (management)team worstelen en weet je dathet anders moet? Dankzij jouw aanpak worden mensen geprikkeld en gestimuleerd om in verandering te komen.
Waar kom je in terecht?
Fatragis een trainingsbureau in Breda met eigen mensen in dienst en webestaan albijna 40jaar. Weonderscheiden onsdoor mensen met ambitie en talent aan te nemen, ze op te leiden en uit te dagen uit hun comfortzone te stappen zodat zij snel het verschil gaan maken bij onze klanten.Écht impact maken en mensen in beweging brengen, dat is waar wij enthousiast van worden!
Wat ga je allemaal doen en leren?
Je gaat de vloer op en observeert scherp hoe er met elkaar en de klant, inwoner of patiënt wordt omgegaan.
Je spiegelt gedrag, geeft eerlijke feedback en deelt je mening, ook als de waarheid soms lastig is.
Je werkt nauw samen met teams en organisaties om veilige en fijne werkklimaten te creëren.
Je daagt mensen uit om actief deel te nemen aan hun eigen ontwikkelingsproces. Geen powerpoint sessies, maar direct in de praktijk doen wat nodig is.
Onze ideale nieuwe collega:
Iemand die af wil van eindeloze, theoretische trainingen en direct impact wil maken op de werkvloer.
Je hebt ervaring met het faciliteren van gedragsverandering en weet hoe lastig dat kan zijn. Niet per se als trainer, maar misschien door je functie of juist in andere situaties?
Je gelooft in het potentieel van mensen en bent in staat dat te stretchen en uit te dagen.
Je bent pragmatisch ingesteld: je houdt van aanpakken en je bent niet bang om de waarheid te vertellen, zelfs als die niet altijd welkom is.
Je bent2 tot 3 dagen per week beschikbaar als trainer, je bent het verlengstuk van onze accountmanagers, je signaleert commerciële kansen en krijgt hier ook energie van
Je bent op zoek naar een vaste aanstelling, we zijn niet op zoek naar ZZPérs
Ten slotte nog even dit…
Naast dagelijks bezig zijn met je eigen ontwikkeling hebben wij ook nog 2 keer per jaar een opleidingsweek met het hele team;
De doorgroeimogelijkheden binnen Fatrag zijn eindeloos,als jij een ambitie hebt dan gaan wij er samen voor om die te behalen;
Een vast salaris dat past bij je ervaring en ambitie
Een mogelijkheid tot een auto van de zaak
Een volledige opleiding tot het worden van een van de beste trainers en coaches van het land.
De kans om samen te werken aan een werkwereld waar echte verbinding en samenwerking de norm zijn.
We gaan ervoor je verwachtingen te overtreffen en nemen 100% verantwoordelijkheid voor onze 50%. Wij zijn er voor je, bij elke stap – we laten je niet zwemmen.
Wil jij samen met ons de werkvloer opschudden en zorgen voor blijvende, betekenisvolle veranderingen? Solliciteer nu en laat ons weten hoe jij de broccoli in een wereld vol frietjes gaat zijn.
Reageer door je CV& motivatiebriefop te sturen naar:.We horen graag van je hoe jij graag wilt solliciteren en kennismaken met ons bedrijf.Hopelijk tot snel!
        """,

        # Vacature Accountmanager (volledige tekst)
        """
VACATURE
Vacature:Accountmanager.
Als jij gepassioneerd bent om jouwtalenten te benutten, grenzen te verleggen en samen met een toegewijd team impact te maken als excellente accountmanager én trainer, dan is dit de perfecte kans voor jou!
Waar kom je in terecht?
Fatragis een trainingsbureau in Bredamet eigen mensen in diensten webestaan al meer dan 35 jaar. We springen eruit door mensen met ambitie en talent aan te nemen, ze op te leiden en uit te dagen uit hun comfortzone te stappen zodat zij snel het verschil gaan maken bij onze klanten. cht impact maken en mensen in beweging brengen.
Wat ga je allemaal doen en leren?
In je rol als accountmanager betekent dit heel goed worden in acquisitie, je eigen portefeuille ontwikkelen en beheren, en klantrelaties opbouwen en uitbouwen. Je bent actief in uiteenlopende branches zoals bijvoorbeeld overheid en zorg, maar ook techniek, productie of logistiek. Je bent vrij om zelf iets op te bouwen en succesvol te zijn, jij bent in de lead!
Onze ideale nieuwe collega:
Heeft minimaal HBO werk- en denkniveau,;
Denkt en handelt commercieel;
Heeft minimaal 3 jaar ervaring in sales(zou mooi zijn maar hoeft niet);
Is oprecht. In gedrag naar anderen en naarjezelf;
Is 32/40 uur beschikbaar;
Heeft een rijbewijs en is bereid heel (of half ;-) Nederland door te reizen.
Ten slotte nog even dit…
Naast je salaris ontvang je ook een interessante provisieregeling en heb je 24 verlofdagen;
Je krijgt een laptop, telefoon en mag gebruik maken van een autoregeling;
Maandagochtend beginnen we de week gezamenlijk verder is hybride werken geen probleem;
Naast dagelijks bezig zijn met je eigen ontwikkeling hebben wij ook nog 2 keer per jaar een opleidingsweek met het hele team;
Écht een unieke kans om in een zeer korte tijd opgeleid te worden tot de beste accountmanager én trainer op het gebied van gedrag en communicatie.
Zin in? Reageer dan door je CV op te sturen naar:.Een motivatiebrief is niet nodig, we zien je liever gewoon ‘live’. We horen graag van je hoe jij graag wilt solliciteren en kennismaken met ons bedrijf. De eerste ‘ronde’ organiseren wij, daarna is het aan jou…
FRANSE AKKER 4, 4824AK BREDA+31 (0)76 571 2930 |
  Fatrag  Fatrag Partner
        """,

        # Missie, ambitie, beloften (volledige tekst)
        """
Wij zijn de broccoli in een wereld vol frietjes. 
Het is onze missie om de status quo van organisatie- en gedragsverandering op z'n kop te zetten met onze unieke Next Step Aanpak. 

We onderzoeken door de vloer op te gaan en scherp vast te leggen hoe er met elkaar en de klant, inwoner of patiënt wordt omgegaan. We spiegelen jullie gedrag en geven onze mening en ons advies. Daarna gaan we aan de slag.
Wij hebben de ervaring en onze hele rugzak vol ideeën, theorieën en praktische methodes die werken, maar jullie zijn in de lead. We prikkelen, zijn soms scherp en vertellen je de waarheid, ook als je daar misschien niet op zit te wachten. Broccoli in plaats van frietjes noemen wij dat.
Maar we stellen je ook gerust, we zijn je grootste supporter en steun en je mag ons ’s nachts bellen. (al moeten we het dan ook even over je timemanagement hebben).
Wat maakt onze aanpak zo anders?
• Geen powerpoint sessies en theoretische trainingen waar iedereen ‘doorheen moet’, maar doen wat nodig is. Op de vloer, in de praktijk
• We zetten iedereen ‘AAN’ om dingen zelf te doen
• We doen het voor, dan doen we het samen en daarna kunnen jullie het zelf
• We dagen je uit en stretchen je potentieel
• We beginnen klein en bereiken uiteindelijk je gehele team, afdeling en/of organisatie
Wat beloven we?
• Meer energie, trots en drive
• We gaan ervoor je verwachtingen te overtreffen
• We gaan door, ook als het moeilijk wordt
• We nemen 100% verantwoordelijkheid voor onze 50%
• Wij zijn er voor je, bij iedere stap, we laten je niet zwemmen
Wanneer moet je ons niet hebben?
• Als je de waarheid liever niet wilt horen
• Als je niet wilt worden uitgedaagd
• Als je liever geen reuring creëert
• Als je een snelle oplossing wil
        """,

        # Next Step PDF (volledige geconsolideerde extractie)
        """
Aanpak DE NEXT STEP SINCERELY FATRAG
Ambitie Advies Activatie Aansluiten Afmaken
DE NEXT STEP STEP 1 STEP 2 STEP 3 STEP 4 STEP 5
STEP 1 Ambitie VOORDOEN SAMEN DOEN ZELF DOEN
Probleem vs. ambitie
Doel: Gewenst gedrag
Commitment 50% - 50% verantwoordelijkheid
Wil je dit echt? We zijn al begonnen
METEN VERSLAGLEGGING
AANGESLOTEN < 5%
Zelf niet willen veranderen
STEP 2 Advies VOORDOEN SAMEN DOEN ZELF DOEN
Uitvoer scan
Plan van aanpak
Delen & bespreken scanverslag
Objectieve, eerlijke spiegel van de situatie
Een realistisch advies
Het delen van het verslag kan reuring opleveren
METEN VERSLAGLEGGING
AANGESLOTEN 5-10%
Er kan weerstand komen op de aankondiging en onze aanwezigheid
STEP 3 Activatie VOORDOEN SAMEN DOEN ZELF DOEN
Kerngroep oprichten
Stuurgroep oprichten
Managementteam sessies
Eerste interventies vinden plaats
Klein beginnen en vanuit positiviteit op alle lagen
Focus ligt op ieders eigen gedrag
Andere aanpak dan men gewend is
METEN VERSLAGLEGGING
AANGESLOTEN 10-20%
STEP 4 Aansluiten VOORDOEN SAMEN DOEN ZELF DOEN
Kerngroepplan + meeting kerngroep
Vervolggroepen
Interventiekeuzes
Oprichten werkgroepen
Train de trainer
We gaan door, ook als het moeilijk wordt
Geen standaard programma's, doen wat nodig is
METEN VERSLAGLEGGING
AANGESLOTEN 20-80%
'Bescherm' het kernteam • Wisseling in MT • Interne politiek
STEP 5 Afmaken VOORDOEN SAMEN DOEN ZELF DOEN
Integratie in organisatie/afdeling processen
Managementcyclus | Coachcyclus | HR | Eigen trainingen
Borgingplan
Audit na x periode
We laten zo snel mogelijk los als het kan
AANGESLOTEN > 80%
Te snel denken dat we er al zijn
'Kies hier een vaste quote' Let's do this! SINCERELY FATRAG
        """,
    ]

    # Combine provided + scraped site data
    scraped_docs = load_scraped("./fatrag_data")
    all_docs = provided_docs + scraped_docs

    # Chunking voor RAG (500 tokens, overlap 100) met compatibele API
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    if hasattr(text_splitter, "split_texts"):
        chunks = text_splitter.split_texts(all_docs)
    else:
        chunks = []
        for t in all_docs:
            chunks.extend(text_splitter.split_text(t))
    return chunks

if __name__ == "__main__":
    chunks = build_chunks()
    print(f"Created {len(chunks)} chunks.")
    # Optioneel: save chunks voor inspectie
    with open("chunks.txt", "w", encoding="utf-8") as f:
        f.write("\n\n".join(chunks))
