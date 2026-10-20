const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const fa = require("react-icons/fa");
// Writes THEME's colours into the .pptx theme part (pptxgenjs cannot); see README.md.
if (!process.env.APPLY_THEME) throw new Error("set APPLY_THEME to the path of apply_theme.js (see README.md)");
const { applyTheme } = require(process.env.APPLY_THEME);

const OUT = __dirname + "/pymarxan_saugomos_teritorijos.pptx";

const THEME = {
  name: "Lagoon",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "16323A", lt1: "FFFFFF", dk2: "0F4C5C", lt2: "EEF3F2",
    accent1: "2E8B7A", accent2: "C8553D", accent3: "E0A93B",
    accent4: "6C8A99", accent5: "9BC1B4", accent6: "3D6B8C",
    hlink: "2E8B7A", folHlink: "6C8A99",
  },
};
const H = THEME.colors;

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5
pres.title = "pymarxan: saugomų teritorijų planavimas";
pres.author = "Klaipėdos universitetas";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;

// ---------- layouts ----------
pres.defineSlideMaster({
  title: "TITLE_DARK",
  background: { color: C.text2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 2.0, w: 11.7, h: 2.0,
      fontFace: THEME.headFontFace, fontSize: 44, bold: true, color: C.background1, valign: "bottom", align: "left" }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 4.2, w: 11.7, h: 1.4,
      fontSize: 20, color: C.accent5, valign: "top" }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "SECTION_DARK",
  background: { color: C.text2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 2.6, w: 11.7, h: 1.3,
      fontFace: THEME.headFontFace, fontSize: 40, bold: true, color: C.background1, valign: "bottom", align: "left" }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 4.0, w: 11.7, h: 1.0,
      fontSize: 20, color: C.accent5, valign: "top" }, text: "" } },
  ],
  slideNumber: { x: 12.3, y: 6.95, w: 0.6, h: 0.3, fontSize: 10, color: C.accent5, align: "right" },
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: C.background1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 12.1, h: 0.85,
      fontFace: THEME.headFontFace, fontSize: 32, bold: true, color: C.text1, valign: "middle", align: "left", margin: 0 }, text: "" } },
    { text: { text: "pymarxan · saugomų teritorijų planavimas", options: {
      x: 0.6, y: 6.95, w: 6, h: 0.3, fontSize: 10, color: C.accent4, margin: 0 } } },
  ],
  slideNumber: { x: 12.1, y: 6.95, w: 0.6, h: 0.3, fontSize: 10, color: C.accent4, align: "right" },
});

// ---------- helpers ----------
async function icon(name, hex = "FFFFFF") {
  const svg = ReactDOMServer.renderToStaticMarkup(React.createElement(fa[name], { color: "#" + hex, size: 256 }));
  const buf = await sharp(Buffer.from(svg)).resize(256, 256).png().toBuffer();
  return "image/png;base64," + buf.toString("base64");
}
async function iconCircle(slide, name, x, y, d, fill, id) {
  slide.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: fill }, line: { type: "none" }, objectName: id + "-circle" });
  const pad = d * 0.24;
  slide.addImage({ data: await icon(name), x: x + pad, y: y + pad, w: d - 2 * pad, h: d - 2 * pad, objectName: id + "-icon" });
}
function card(slide, x, y, w, h, id, fill = C.background2) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.12, fill: { color: fill }, line: { type: "none" }, objectName: id });
}
function txt(slide, text, o) {
  slide.addText(text, { isTextBox: true, margin: 0, valign: "top", color: C.text1, fontSize: 14, ...o });
}
const content = (section) => pres.addSlide({ masterName: "CONTENT", sectionTitle: section });

(async () => {
  // ===== 1. Title =====
  pres.addSection({ title: "Įvadas" });
  let s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Įvadas" });
  s.addText("pymarxan: duomenimis grįstas saugomų teritorijų planavimas", { placeholder: "title" });
  s.addText("Kūrimo būklė, taikymas praktikoje ir dar neišplėtotos galimybės\nArtūras Razinkovas-Baziukas · Klaipėdos universitetas · 2026 m. spalis", { placeholder: "body" });
  await iconCircle(s, "FaMapMarkedAlt", 0.8, 0.8, 0.9, C.accent1, "title");
  s.addNotes("Šiandien pristatysiu pymarxan – atvirąją programinę įrangą saugomų teritorijų planavimui. Aptarsime, kokius sprendimus ji padeda priimti, kiek ji yra išbaigta šiandien ir – svarbiausia – ko joje dar trūksta, kad ją būtų galima naudoti kasdieniame administracijų darbe.");

  // ===== 2. Decisions =====
  s = content("Įvadas");
  s.addText("Sprendimai, kuriuos tenka priimti", { placeholder: "title" });
  const decisions = [
    ["FaMapMarkedAlt", "Plėsti tinklą", "Kurias teritorijas saugoti, kad pasiektume 30 % tikslą mažiausiomis sąnaudomis?"],
    ["FaLayerGroup", "Zonuoti teritoriją", "Kur griežta apsauga, kur tausojantis naudojimas, kur atkūrimas?"],
    ["FaFish", "Atkurti upes", "Kurias užtvankas ar slenksčius šalinti, kad žuvys pasiektų daugiausia buveinių?"],
    ["FaSeedling", "Skirstyti atkūrimo lėšas", "Kur investuoti, kad atkurtos buveinės susijungtų į vientisus plotus?"],
    ["FaCloudSunRain", "Valdyti riziką", "Kaip atsižvelgti į buveinių praradimo riziką ir klimato kaitą?"],
  ];
  for (let i = 0; i < decisions.length; i++) {
    const [ic, head, body] = decisions[i];
    const x = 0.6 + i * 2.48, y = 1.6, w = 2.2, h = 3.75;
    card(s, x, y, w, h, "dec-card-" + i);
    await iconCircle(s, ic, x + 0.25, y + 0.3, 0.8, C.accent1, "dec-" + i);
    txt(s, head, { x: x + 0.25, y: y + 1.3, w: w - 0.5, h: 0.7, fontSize: 18, bold: true, fontFace: THEME.headFontFace, valign: "top" });
    txt(s, body, { x: x + 0.25, y: y + 2.05, w: w - 0.5, h: 1.7, fontSize: 14 });
  }
  txt(s, "Visiems klausimams bendra: ribotas biudžetas, daug saugotinų vertybių ir aiškiai suformuluoti tikslai.",
    { x: 0.6, y: 5.85, w: 12.1, h: 0.5, fontSize: 16, italic: true, color: C.text2 });
  s.addNotes("Tai penki tipiniai klausimai, su kuriais susiduria saugomų teritorijų administracijos. Visi jie turi tą pačią struktūrą: yra daug galimų vietų, ribotas biudžetas ir aiškūs tikslai – pavyzdžiui, ES biodiversiteto strategijos 30 % tikslas. Būtent tokius uždavinius sprendžia sisteminio apsaugos planavimo įrankiai. Toliau kiekvieną skaidrę siesiu su vienu iš šių klausimų.");

  // ===== 3. How it works =====
  s = content("Įvadas");
  s.addText("Kaip veikia pymarxan", { placeholder: "title" });
  const steps = [
    ["FaDatabase", "Duomenys", "Planavimo vienetai, buveinių ir rūšių paplitimas, sąnaudos ar spaudimas, esamos saugomos teritorijos"],
    ["FaBullseye", "Tikslai", "Kiek kiekvienos vertybės išsaugoti, kas jau saugoma, ko negalima įtraukti"],
    ["FaCogs", "Optimizavimas", "Algoritmas peržiūri labai daug derinių ir randa pigiausią visus tikslus tenkinantį variantą"],
    ["FaMap", "Rezultatai", "Žemėlapiai, tikslų įvykdymo lentelės, alternatyvos ir kiekvieno ploto svarba"],
  ];
  for (let i = 0; i < steps.length; i++) {
    const [ic, head, body] = steps[i];
    const x = 0.6 + i * 3.1, y = 1.55, w = 2.7, h = 3.25;
    card(s, x, y, w, h, "step-card-" + i);
    await iconCircle(s, ic, x + 0.25, y + 0.25, 0.75, i === 2 ? C.accent2 : C.accent1, "step-" + i);
    txt(s, head, { x: x + 0.25, y: y + 1.15, w: w - 0.5, h: 0.5, fontSize: 18, bold: true, fontFace: THEME.headFontFace });
    txt(s, body, { x: x + 0.25, y: y + 1.7, w: w - 0.5, h: 1.45, fontSize: 14 });
    if (i < steps.length - 1)
      s.addShape(pres.shapes.RIGHT_ARROW, { x: x + w + 0.08, y: y + 1.45, w: 0.24, h: 0.35, fill: { color: C.accent4 }, line: { type: "none" }, objectName: "step-arrow-" + i });
  }
  card(s, 0.6, 5.15, 12.1, 1.45, "marxan-note", C.text2);
  txt(s, [
    { text: "Marxan", options: { bold: true } },
    { text: " – plačiausiai pasaulyje naudojama saugomų teritorijų planavimo programa. " },
    { text: "pymarxan", options: { bold: true } },
    { text: " – atvirasis jos atkūrimas Python kalba, papildytas tiksliaisiais sprendikliais, zonavimu, jungtumo ir upių moduliais bei internetine programa." },
  ], { x: 0.9, y: 5.3, w: 11.5, h: 1.15, fontSize: 16, color: C.background1, valign: "middle" });
  s.addNotes("Darbo eiga visada ta pati. Pirmiausia teritorija suskaidoma į planavimo vienetus – tai gali būti sklypai arba rastro langeliai. Kiekvienam vienetui žinome, kiek jame yra kiekvienos saugotinos vertybės ir kiek kainuotų jį saugoti. Tada nustatome tikslus, o algoritmas suranda pigiausią derinį, kuris visus juos tenkina. Marxan yra pasaulinis standartas šioje srityje; pymarxan jį atkuria ir gerokai išplečia.");

  // ===== 4. Status =====
  pres.addSection({ title: "Būklė ir galimybės" });
  s = content("Būklė ir galimybės");
  s.addText("Kūrimo būklė (2026 m. spalis)", { placeholder: "title" });
  const stats = [["45", "versijos per 5 mėnesius\n(v0.1 → v0.45)"], ["3 700+", "automatinių testų"], ["~33 000", "programos kodo eilučių"], ["11", "internetinės programos skilčių"]];
  for (let i = 0; i < stats.length; i++) {
    const x = 0.6 + i * 3.1;
    txt(s, stats[i][0], { x, y: 1.45, w: 2.8, h: 1.0, fontSize: 54, bold: true, color: C.accent1, fontFace: THEME.headFontFace, valign: "bottom" });
    txt(s, stats[i][1], { x, y: 2.5, w: 2.8, h: 0.75, fontSize: 14, color: C.accent4 });
  }
  card(s, 0.6, 3.55, 5.9, 3.05, "ready-card");
  card(s, 6.8, 3.55, 5.9, 3.05, "missing-card");
  await iconCircle(s, "FaCheckCircle", 0.85, 3.8, 0.55, C.accent1, "ready");
  await iconCircle(s, "FaExclamationTriangle", 7.05, 3.8, 0.55, C.accent2, "missing");
  txt(s, "Kas paruošta", { x: 1.6, y: 3.85, w: 4.6, h: 0.45, fontSize: 18, bold: true, fontFace: THEME.headFontFace, valign: "middle" });
  txt(s, "Ko trūksta platinimui", { x: 7.8, y: 3.85, w: 4.6, h: 0.45, fontSize: 18, bold: true, fontFace: THEME.headFontFace, valign: "middle" });
  const bl = (arr) => arr.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < arr.length - 1, paraSpaceAfter: 6 } }));
  txt(s, bl(["Visa Marxan šeima: Marxan, Marxan with Zones, Marxan with Connectivity",
    "Tikslieji sprendikliai su optimalumo garantija",
    "Internetinė programa, nereikalaujanti programavimo",
    "Kiekvienas pakeitimas automatiškai tikrinamas"]), { x: 0.85, y: 4.5, w: 5.4, h: 2.0 });
  txt(s, bl(["Neplatinama per PyPI – diegiama iš GitHub",
    "Sąsaja ir pagalba – tik anglų kalba",
    "Išbandyta su viešais Marxan duomenų rinkiniais ir sintetiniais scenarijais, dar ne su administracijos duomenimis"]), { x: 7.05, y: 4.5, w: 5.4, h: 2.0 });
  s.addNotes("Projektas aktyviai vystomas nuo 2026 m. gegužės: per penkis mėnesius išleistos 45 versijos, dabartinė – 0.45.0. Daugiau nei 3 700 automatinių testų paleidžiama po kiekvieno pakeitimo. Programinė įranga atvira, MIT licencija, laikoma GitHub. Svarbu sąžiningai pasakyti: tai dar ne „dėžutės\" produktas – jis neplatinamas per standartinį Python paketų katalogą, sąsaja angliška, o realaus projekto su administracijos duomenimis dar nebuvo.");

  // ===== 5. Capabilities =====
  s = content("Būklė ir galimybės");
  s.addText("Pagrindinės galimybės", { placeholder: "title" });
  const caps = [
    ["FaMapMarkedAlt", "Teritorijų atranka", "Pigiausias tinklas, tenkinantis visus tikslus; kompaktiškumo svertas (BLM) mažina fragmentaciją."],
    ["FaLayerGroup", "Zonavimas", "Kelios valdymo zonos su skirtingomis sąnaudomis ir indėliu į tikslus (Marxan with Zones)."],
    ["FaProjectDiagram", "Jungtumas ir klimatas", "Ekologiniai koridoriai, grandinių teorija, klimato kaitos greitis ir refugijos."],
    ["FaWater", "Upių jungtumas", "Kuriuos barjerus šalinti už turimą biudžetą (DCI indeksas, įrodytas optimumas)."],
    ["FaChartBar", "Prioritetų žemėlapis", "Zonation tipo tęstinis prioritetų reitingas visai teritorijai, net milijonui langelių."],
    ["FaBalanceScale", "Ataskaitos ir politika", "30×30 vertinimas, spragų analizė, naštos pasiskirstymas, atsparumas scenarijams, WDPA importas."],
  ];
  for (let i = 0; i < caps.length; i++) {
    const [ic, head, body] = caps[i];
    const col = i % 3, row = Math.floor(i / 3);
    const x = 0.6 + col * 4.1, y = 1.55 + row * 2.6, w = 3.8, h = 2.3;
    card(s, x, y, w, h, "cap-card-" + i);
    await iconCircle(s, ic, x + 0.25, y + 0.3, 0.7, C.accent1, "cap-" + i);
    txt(s, head, { x: x + 1.15, y: y + 0.3, w: w - 1.35, h: 0.7, fontSize: 18, bold: true, fontFace: THEME.headFontFace, valign: "middle" });
    txt(s, body, { x: x + 0.25, y: y + 1.15, w: w - 0.5, h: 1.05, fontSize: 14 });
  }
  s.addNotes("Šešios pagrindinės galimybių grupės. Teritorijų atranka ir zonavimas – tai klasikinis Marxan. Jungtumo modulis leidžia įvertinti ekologinius koridorius ir klimato refugijas. Upių modulis sprendžia, kuriuos barjerus šalinti. Prioritetų žemėlapis veikia kaip Zonation – sureitinguoja visą teritoriją. Ataskaitos padeda atsiskaityti už 30×30 tikslus; galima tiesiogiai importuoti esamas saugomas teritorijas iš Protected Planet (WDPA) duomenų bazės.");

  // ===== 6. Case study map =====
  pres.addSection({ title: "Taikymas" });
  s = content("Taikymas");
  s.addText("Pavyzdys: Kuršių marios ir Klaipėdos priekrantė", { placeholder: "title" });
  s.addImage({ path: __dirname + "/baltic_map.png", x: 0.9, y: 1.3, w: 11.5, h: 11.5 * 960 / 2700, objectName: "baltic-map" });
  const cs = [["4 vertybės, 256 vienetai", "nendrynai, makrofitai, starkių nerštavietės, vandens paukščiai"],
    ["Tikslas – po 30 %", "kiekvienos vertybės regione; 2 esamos ST užfiksuotos"],
    ["24 % pigiau", "tikslusis sprendimas (49,7) prieš klasikinę euristiką (65,1)"]];
  for (let i = 0; i < cs.length; i++) {
    const x = 0.6 + i * 4.1;
    txt(s, cs[i][0], { x, y: 5.5, w: 3.8, h: 0.45, fontSize: 18, bold: true, color: i === 2 ? C.accent2 : C.text2, fontFace: THEME.headFontFace });
    txt(s, cs[i][1], { x, y: 5.95, w: 3.8, h: 0.6, fontSize: 14, color: C.text1 });
  }
  txt(s, "Sintetinis demonstracinis scenarijus (examples/baltic_marine_planning.py)", { x: 6.7, y: 6.95, w: 5.2, h: 0.3, fontSize: 10, color: C.accent4, align: "right" });
  s.addNotes("Konkretus pavyzdys mūsų regione. Tai sintetinis, bet realistiškos struktūros scenarijus: Kuršių marios ir priekrantė suskaidytos į 256 langelius, sąnaudos – žmogaus veiklos spaudimas, didžiausias prie uosto. Turime keturias vertybes ir dvi jau esamas saugomas teritorijas. Kairėje – sąnaudų žemėlapis, viduryje – tikslusis optimalus sprendimas, dešinėje – klasikinio Marxan algoritmo rezultatas. Abu tenkina visus tikslus, bet tikslusis sprendimas yra 24 % pigesnis ir kompaktiškesnis. Tai ir yra viena iš pymarxan pridėtinių verčių: galima gauti įrodytą optimumą, o ne tik gerą apytikslį sprendimą.");

  // ===== 7. Gap analysis chart =====
  s = content("Taikymas");
  s.addText("Spragų analizė: prieš ir po", { placeholder: "title" });
  s.addChart(pres.charts.BAR, [
    { name: "Esamos ST", labels: ["Nendrynai", "Povandeniniai makrofitai", "Starkių nerštavietės", "Migruojantys vandens paukščiai"], values: [2.4, 0.3, 0.0, 1.5] },
    { name: "Siūlomas tinklas", labels: ["Nendrynai", "Povandeniniai makrofitai", "Starkių nerštavietės", "Migruojantys vandens paukščiai"], values: [30.4, 30.0, 86.0, 30.2] },
  ], {
    x: 0.6, y: 1.4, w: 7.6, h: 5.2, barDir: "bar", barGrouping: "clustered", barGapWidthPct: 60,
    chartColors: [H.accent4, H.accent1],
    showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 12, dataLabelFontFace: "+mn-lt", dataLabelColor: H.dk1, dataLabelFormatCode: "0.0",
    catAxisLabelColor: H.dk1, catAxisLabelFontSize: 13, catAxisLabelFontFace: "+mn-lt", catAxisOrientation: "maxMin",
    valAxisLabelColor: H.accent4, valAxisLabelFontSize: 11, valAxisLabelFontFace: "+mn-lt", valAxisMaxVal: 100, valAxisMinVal: 0,
    valGridLine: { color: "DDE5E3", size: 0.5 }, catGridLine: { style: "none" },
    showLegend: true, legendPos: "b", legendFontSize: 12, legendFontFace: "+mn-lt", legendColor: H.dk1,
    showTitle: true, title: "Apsaugota vertybės dalis, %", titleFontSize: 14, titleColor: H.dk1, titleFontFace: "+mn-lt",
    objectName: "gap-chart",
  });
  const gap = [["0–2,4 %", "tiek kiekvienos vertybės apima esamos saugomos teritorijos"],
    ["≥ 30 %", "pasiekia siūlomas tinklas visoms keturioms vertybėms"],
    ["86 %", "starkių nerštavietės apsaugomos „pakeliui“ – kartu su kitomis vertybėmis"]];
  for (let i = 0; i < gap.length; i++) {
    const y = 1.55 + i * 1.7;
    txt(s, gap[i][0], { x: 8.7, y, w: 4.0, h: 0.7, fontSize: 36, bold: true, color: i === 0 ? C.accent2 : C.accent1, fontFace: THEME.headFontFace, valign: "bottom" });
    txt(s, gap[i][1], { x: 8.7, y: y + 0.75, w: 4.0, h: 0.8, fontSize: 14 });
  }
  s.addNotes("Spragų analizė parodo, kiek kiekvienos vertybės apima esamas tinklas ir kiek – siūlomas. Šiame pavyzdyje esamos teritorijos apima vos kelis procentus, o siūlomas tinklas pasiekia 30 % tikslą visoms vertybėms. Tokia lentelė – tiesioginis argumentas ataskaitose ir derybose: matyti, kurių vertybių apsauga nepakankama ir kiek trūksta.");

  // ===== 7b. Reed-bed mosaic (examples/curonian_spit_reedbeds.py via mosaic.json) =====
  const MOSAIC = JSON.parse(require("fs").readFileSync(__dirname + "/mosaic.json", "utf8"));
  const lt = (v, d = 1) => v.toFixed(d).replace(".", ",");
  s = content("Taikymas");
  s.addText("Zonavimas: nendrynų mozaika Kuršių nerijoje", { placeholder: "title" });
  txt(s, "Nendrynų įvairovė didžiausia, kai pjaunami ir nepjaunami ruožai kaitaliojasi. Kiekviena jų riba gauna premiją, todėl planas skaido nendryną, o ne jungia jį į ištisus plotus.",
    { x: 0.6, y: 1.3, w: 12.1, h: 0.7, fontSize: 16 });
  const nSeg = MOSAIC.n_segments, cellW = 12.1 / nSeg;
  const strips = [["Pigiausias planas (be premijos)", MOSAIC.plain.pattern, 2.25],
    [`Mozaika (premija ${lt(MOSAIC.mosaic.edge_reward)} k€ už kiekvieną ribą)`, MOSAIC.mosaic.pattern, 3.35]];
  for (const [label, pattern, y] of strips) {
    txt(s, label, { x: 0.6, y, w: 8, h: 0.35, fontSize: 14, bold: true, color: C.text2 });
    [...pattern].forEach((ch, k) => s.addShape(pres.shapes.RECTANGLE, {
      x: 0.6 + k * cellW, y: y + 0.4, w: cellW - 0.025, h: 0.5,
      fill: { color: ch === "C" ? C.accent3 : C.accent1 }, line: { type: "none" },
      objectName: `mosaic-${y}-${k}`,
    }));
  }
  const align = { "Pervalka": "right", "Preila": "left", "Smiltynė": "left" };
  for (const [name, km] of Object.entries(MOSAIC.settlements)) {
    const a = align[name] || "center", x = 0.6 + km * cellW;
    const bx = a === "left" ? x : a === "right" ? x + cellW - 1.4 : x + cellW / 2 - 0.7;
    txt(s, name, { x: bx, y: 4.32, w: 1.4, h: 0.3, fontSize: 12, color: C.accent4, align: a });
  }
  txt(s, "Smiltynė (šiaurė) → siena (pietūs), 1 langelis = 1 km", { x: 7.2, y: 4.76, w: 5.5, h: 0.33, fontSize: 12, color: C.accent4, align: "right", valign: "middle" });
  [[C.accent3, "Pjaunama šį sezoną"], [C.accent1, "Paliekama (sena nendrė)"]].forEach(([col, lab], i) => {
    s.addShape(pres.shapes.RECTANGLE, { x: 0.6 + i * 3.2, y: 4.8, w: 0.25, h: 0.25, fill: { color: col }, line: { type: "none" }, objectName: "mosaic-legend-" + i });
    txt(s, lab, { x: 0.95 + i * 3.2, y: 4.76, w: 2.8, h: 0.33, fontSize: 14, valign: "middle" });
  });
  const P = MOSAIC.plain, M = MOSAIC.mosaic;
  const mstats = [[`${P.edges} → ${M.edges}`, "ribų tarp pjaunamų ir nepjaunamų ruožų"],
    [`${P.longest_uncut_km} → ${M.longest_uncut_km} km`, "ilgiausias vientisas nepjaunamas ruožas"],
    [`+${lt(M.cutting_cost - P.cutting_cost)} k€`, "papildomos pjovimo išlaidos už mozaiką"]];
  mstats.forEach(([big, small], i) => {
    const x = 0.6 + i * 4.1;
    txt(s, big, { x, y: 5.35, w: 3.8, h: 0.75, fontSize: 36, bold: true, color: i === 2 ? C.accent2 : C.accent1, fontFace: THEME.headFontFace, valign: "bottom" });
    txt(s, small, { x, y: 6.15, w: 3.8, h: 0.5, fontSize: 14 });
  });
  txt(s, "Sintetinis scenarijus (examples/curonian_spit_reedbeds.py)", { x: 6.7, y: 6.95, w: 5.2, h: 0.3, fontSize: 10, color: C.accent4, align: "right" });
  s.addNotes(`Antras pavyzdys iš mūsų regiono – nendrynai Kuršių nerijos marių pakrantėje. Čia uždavinys priešingas klasikiniam Marxan: nendrynų biologinei įvairovei reikia ne vientisų plotų, o mozaikos, kurioje pjaunami ir nepjaunami ruožai kaitaliojasi. Pakrantę nuo Smiltynės iki sienos padalijome į ${nSeg} vieno kilometro ruožus; kiekvienas šį sezoną arba pjaunamas, arba paliekamas. Senos nendrės buveinė skaičiuojama tik nepjautuose ruožuose, jaunos nendrės mozaikos buveinė – tik pjautuose. Viršutinė juosta – pigiausias planas: pjaunama keliais ištisais blokais prie gyvenviečių. Apatinė – kai kiekviena pjauto ir nepjauto ruožo riba gauna premiją: ribų padaugėja nuo ${P.edges} iki ${M.edges}, o ilgiausias vientisas nepjautas ruožas sutrumpėja nuo ${P.longest_uncut_km} iki ${M.longest_uncut_km} km. Tai kainuoja apie ${lt(M.cutting_cost - P.cutting_cost)} tūkst. eurų daugiau. Premijos dydis – valdymo sprendimas: programa parodo, kiek kainuoja kiekviena papildoma riba. Duomenys sintetiniai, gyvenviečių padėtis apytikslė.`);

  // ===== 8. Rivers =====
  s = content("Taikymas");
  s.addText("Upių atkūrimas: kuriuos barjerus šalinti", { placeholder: "title" });
  s.addImage({ path: __dirname + "/river_network_optimization_lt.png", x: 0.6, y: 1.45, w: 7.4, h: 7.4 * 737 / 1290, objectName: "river-map" });
  s.addImage({ path: __dirname + "/dci_budget_frontier_lt.png", x: 8.4, y: 1.45, w: 4.3, h: 4.3 * 584 / 945, objectName: "dci-frontier" });
  txt(s, bl(["Žuvų migracijai svarbus upių tinklo jungtumas (DCI indeksas)",
    "Optimalus barjerų rinkinys už turimą biudžetą",
    "Kreivė rodo, kiek jungtumo nuperka kiekvienas papildomas euras",
    "Įkeliami HydroRIVERS ar nacionaliniai upių duomenys"]), { x: 8.4, y: 4.3, w: 4.3, h: 2.4 });
  s.addNotes("Upių modulis atsako į klausimą, kurias užtvankas, slenksčius ar pralaidas šalinti pirmiausia. Kairėje – tinklas prieš ir po optimizavimo, dešinėje – kreivė, rodanti, kiek jungtumo padidėja didinant biudžetą. Tai padeda pagrįsti, kodėl verta investuoti būtent į tam tikrus objektus, ir parodyti, kur papildomos lėšos duoda mažai naudos.");

  // ===== 9. Workflow =====
  s = content("Taikymas");
  s.addText("Kaip tai atrodytų darbe", { placeholder: "title" });
  const wf = ["Įkelti duomenis: Marxan failai, rastrai, GIS sluoksniai", "Nustatyti tikslus ir esamas saugomas teritorijas", "Paleisti skaičiavimą internetinėje programoje", "Palyginti alternatyvas ir scenarijus", "Eksportuoti žemėlapius ir ataskaitas sprendimams"];
  for (let i = 0; i < wf.length; i++) {
    const y = 1.55 + i * 0.95;
    s.addShape(pres.shapes.OVAL, { x: 0.6, y, w: 0.6, h: 0.6, fill: { color: C.accent1 }, line: { type: "none" }, objectName: "wf-num-" + i });
    txt(s, String(i + 1), { x: 0.6, y, w: 0.6, h: 0.6, fontSize: 18, bold: true, color: C.background1, align: "center", valign: "middle" });
    txt(s, wf[i], { x: 1.45, y, w: 5.3, h: 0.6, fontSize: 16, valign: "middle" });
  }
  txt(s, "Analitikai tą patį gali atlikti Python skriptuose – pakartojamai ir dokumentuotai.", { x: 0.6, y: 6.3, w: 6.2, h: 0.5, fontSize: 14, italic: true, color: C.text2 });
  card(s, 7.2, 1.4, 5.5, 5.35, "app-frame");
  s.addImage({ path: __dirname + "/rivers_tab_crop.png", x: 7.2 + (5.5 - 5.0 * 868 / 1125) / 2, y: 1.58, w: 5.0 * 868 / 1125, h: 5.0, objectName: "app-screenshot" });
  s.addNotes("Darbuotojui nereikia programuoti: viskas vyksta internetinėje programoje naršyklėje. Įkeliami duomenys, nustatomi tikslai, paleidžiamas skaičiavimas, palyginamos alternatyvos ir eksportuojami rezultatai. Dešinėje – programos upių skiltis: biudžetas, rezultatas ir jungtumo kreivė. Analitikai tą pačią logiką gali naudoti per Python, kad analizę būtų galima pakartoti ir dokumentuoti.");

  // ===== 10. Trust =====
  s = content("Taikymas");
  s.addText("Ar galima pasitikėti rezultatais?", { placeholder: "title" });
  const trust = [["FaCheckCircle", "Patikrinta su žinomu atsakymu", "Testinis uždavinys turi rankomis išvestą optimumą – tikslusis sprendiklis privalo jį pasiekti."],
    ["FaCode", "Sutampa su originaliu Marxan", "Baudų, tikslų ir parametrų logika atkartoja originalų Marxan C++ kodą."],
    ["FaFlask", "Automatiniai testai", "3 700+ testų paleidžiama po kiekvieno pakeitimo; dabartinė versija juos visus praeina."],
    ["FaShieldAlt", "Įrodytas optimumas", "Galima gauti tikslų sprendimą ir palyginti jį su klasikiniu Marxan algoritmu."]];
  for (let i = 0; i < trust.length; i++) {
    const y = 1.5 + i * 1.3;
    await iconCircle(s, trust[i][0], 0.6, y, 0.75, C.accent1, "trust-" + i);
    txt(s, trust[i][1], { x: 1.6, y: y - 0.02, w: 6.3, h: 0.4, fontSize: 18, bold: true, fontFace: THEME.headFontFace });
    txt(s, trust[i][2], { x: 1.6, y: y + 0.42, w: 6.3, h: 0.7, fontSize: 14 });
  }
  card(s, 8.6, 1.5, 4.1, 4.85, "trust-stat", C.text2);
  txt(s, "24 %", { x: 8.9, y: 2.2, w: 3.5, h: 1.3, fontSize: 66, bold: true, color: C.accent3, fontFace: THEME.headFontFace, align: "center", valign: "middle" });
  txt(s, "tiek pigesnis tikslusis sprendimas Kuršių marių pavyzdyje, tenkinant tuos pačius tikslus", { x: 8.9, y: 3.7, w: 3.5, h: 1.6, fontSize: 16, color: C.background1, align: "center" });
  s.addNotes("Administracijai svarbiausia – ar rezultatais galima remtis. Turime keturis patikimumo sluoksnius: uždavinį su žinomu teisingu atsakymu, atitikimą originaliam Marxan kodui, tūkstančius automatinių testų ir galimybę gauti įrodytą optimumą. Pastarasis turi ir praktinę vertę – Kuršių marių pavyzdyje tas pats tikslas pasiekiamas 24 % pigiau.");

  // ===== 11. Section: undeveloped =====
  pres.addSection({ title: "Neišplėtotos galimybės" });
  s = pres.addSlide({ masterName: "SECTION_DARK", sectionTitle: "Neišplėtotos galimybės" });
  s.addText("Dar neišplėtotos galimybės", { placeholder: "title" });
  s.addText("Ko dar trūksta ir ką tai reiškia praktikoje", { placeholder: "body" });
  await iconCircle(s, "FaExclamationTriangle", 0.8, 1.3, 0.9, C.accent2, "section-gap");
  s.addNotes("Dabar – apie tai, ko dar nėra. Tai svarbiausia šio pristatymo dalis, nes būtent čia jūsų poreikiai turėtų nulemti tolesnio kūrimo prioritetus. Suskirsčiau trūkumus į dvi grupes: diegimą ir kasdienį naudojimą bei analitines galimybes.");

  // ===== 12 & 13. Gap tables =====
  const STATUS = { "Nepradėta": H.accent2, "Dalinai": H.accent3, "Suprojektuota": H.accent1 };
  function gapTable(slide, rows) {
    const head = ["Galimybė", "Ką tai reiškia administracijai", "Būklė"].map((t) => ({ text: t, options: { bold: true, color: H.lt1, fill: { color: H.dk2 }, fontSize: 14 } }));
    const body = rows.map(([a, b, st], i) => [
      { text: a, options: { bold: true, fontSize: 14, color: H.dk1, fill: { color: i % 2 ? H.lt1 : H.lt2 } } },
      { text: b, options: { fontSize: 14, color: H.dk1, fill: { color: i % 2 ? H.lt1 : H.lt2 } } },
      { text: st, options: { bold: true, fontSize: 14, color: STATUS[st.split(/[ ,]/)[0]], fill: { color: i % 2 ? H.lt1 : H.lt2 } } },
    ]);
    slide.addTable([head, ...body], { x: 0.6, y: 1.4, w: 12.1, colW: [3.6, 6.4, 2.1], rowH: [0.5, 0.8, 0.8, 0.8, 0.8, 0.8], fontFace: THEME.bodyFontFace, valign: "middle", margin: [0.06, 0.12, 0.06, 0.12], border: { type: "none" }, objectName: "gap-table" });
  }
  s = content("Neišplėtotos galimybės");
  s.addText("Neišplėtota: diegimas ir kasdienis naudojimas", { placeholder: "title" });
  gapTable(s, [
    ["Paprastas įdiegimas", "Šiandien diegiama iš GitHub; be IT specialisto sudėtinga", "Nepradėta (PyPI)"],
    ["Lietuviška sąsaja", "Visa sąsaja ir pagalbos tekstai – anglų kalba", "Nepradėta"],
    ["Bendras darbas įstaigoje", "Programą galima paleisti serveryje, bet nėra prisijungimo, naudotojų teisių ir projektų dalijimosi", "Nepradėta"],
    ["Tikslo pasirinkimas programoje", "Pvz., „daugiausia apsaugos už fiksuotą biudžetą“: valdiklis yra, bet skaičiavimui neperduodamas – veikia tik per Python", "Dalinai"],
    ["Sklaidos nustatymai prioritetų žemėlapiui", "Rūšių sklaidos glodinimas pasiekiamas tik per Python", "Dalinai"],
  ]);
  s.addNotes("Pirmoji grupė – tai, kas trukdo įrankį naudoti be programuotojo. Įdiegimas kol kas reikalauja techninių žinių. Sąsaja angliška. Nėra naudotojų paskyrų ir projektų dalijimosi. Kai kurios galimybės, pavyzdžiui, kitokio tikslo pasirinkimas – kai norime ne pigiausio tinklo, o daugiausia apsaugos už fiksuotą biudžetą – jau veikia programiniame branduolyje, bet dar neprijungtos prie internetinės programos.");

  s = content("Neišplėtotos galimybės");
  s.addText("Neišplėtota: analitinės galimybės", { placeholder: "title" });
  gapTable(s, [
    ["Buveinės praradimo rizika (Marxan Probability 1D)", "Atsižvelgti į tikimybę, kad vieta bus prarasta, pvz., dėl erozijos ar užstatymo", "Suprojektuota, v0.46"],
    ["Rizika zonavimo uždaviniuose", "Zonų sprendikliai kol kas neatsižvelgia į tikimybių lentelę", "Nepradėta"],
    ["Didelio masto atkūrimo planavimas", "Tik apytikslis algoritmas iki kelių tūkstančių langelių; nėra IIC/PC jungtumo indeksų ir buveinių būklių modelio", "Dalinai"],
    ["Jungtumas greituosiuose algoritmuose", "Godieji algoritmai jungtumą tik apskaičiuoja, bet neoptimizuoja", "Dalinai"],
    ["Zonation patobulinimai", "Buveinių būklės sluoksniai, rūšių jautrumo parametras, kaimynystės apribojimas", "Nepradėta"],
  ]);
  txt(s, "Smulkesni darbai: sprendinių rinkiniai, mokomųjų duomenų generatorius, kaimynystės ir linijinės baudos, filogenetinis endemizmas, mažesnis atminties poreikis dideliems rastrams su zonomis.",
    { x: 0.6, y: 6.2, w: 12.1, h: 0.6, fontSize: 12, italic: true, color: C.accent4 });
  s.addNotes("Antroji grupė – analitinės galimybės. Artimiausias darbas – buveinės praradimo rizika: jau suprojektuota ir planuojama kitoje versijoje. Zonavimo uždaviniuose rizika dar nevertinama. Atkūrimo planavimui yra tik greitas apytikslis algoritmas ir vienas suskaidymo rodiklis; trūksta tinklo jungtumo indeksų ir buveinių būklių modelio. Taip pat yra eilė smulkesnių patobulinimų. Kurie iš jų jums svarbiausi – būtent tai norėtume išgirsti.");

  // ===== 14. Roadmap =====
  pres.addSection({ title: "Toliau" });
  s = content("Toliau");
  s.addText("Kelias į priekį", { placeholder: "title" });
  const road = [["Artimiausia versija", "v0.46", ["Buveinės praradimo rizika (Probability 1D)", "Suderinimas su Marxan 3 tikimybių logika"]],
    ["Platinimui", "diegimas", ["Paketas PyPI kataloge", "Lietuviška sąsaja", "Tikslo pasirinkimas internetinėje programoje"]],
    ["Kartu su jumis", "pilotas", ["Bandomasis projektas vienoje teritorijoje", "Tolesnių funkcijų eiliškumas pagal jūsų poreikius"]]];
  s.addShape(pres.shapes.LINE, { x: 1.0, y: 2.05, w: 11.3, h: 0, line: { color: C.accent5, width: 2 }, objectName: "road-line" });
  for (let i = 0; i < road.length; i++) {
    const x = 0.6 + i * 4.1;
    s.addShape(pres.shapes.OVAL, { x: x + 0.1, y: 1.7, w: 0.7, h: 0.7, fill: { color: i === 2 ? C.accent2 : C.accent1 }, line: { color: C.background1, width: 3 }, objectName: "road-dot-" + i });
    txt(s, String(i + 1), { x: x + 0.1, y: 1.7, w: 0.7, h: 0.7, fontSize: 18, bold: true, color: C.background1, align: "center", valign: "middle" });
    card(s, x, 2.75, 3.8, 2.9, "road-card-" + i);
    txt(s, road[i][0], { x: x + 0.3, y: 2.95, w: 3.2, h: 0.45, fontSize: 20, bold: true, fontFace: THEME.headFontFace });
    txt(s, road[i][1], { x: x + 0.3, y: 3.4, w: 3.2, h: 0.4, fontSize: 14, italic: true, color: C.accent4 });
    txt(s, bl(road[i][2]), { x: x + 0.3, y: 3.95, w: 3.2, h: 2.2, fontSize: 14 });
  }
  s.addNotes("Trys žingsniai. Pirmasis jau suplanuotas – buveinės praradimo rizika kitoje versijoje. Antrasis – padaryti įrankį prieinamą be programuotojo: standartinis paketas, lietuviška sąsaja, visų galimybių prijungimas prie internetinės programos. Trečiasis priklauso nuo jūsų – bandomasis projektas vienoje saugomoje teritorijoje, kuris parodytų, ko iš tikrųjų reikia.");

  // ===== 15. Pilot needs =====
  s = content("Toliau");
  s.addText("Ko reikėtų bandomajam projektui", { placeholder: "title" });
  const needs = [["FaDatabase", "Vertybių duomenys", "Buveinių ir rūšių žemėlapiai, Natura 2000 vertybės, inventorizacijos"],
    ["FaSlidersH", "Sąnaudos ir spaudimas", "Žemės vertė, ūkinė veikla, grėsmių ir spaudimo sluoksniai"],
    ["FaShieldAlt", "Taisyklės", "Esamų teritorijų ribos, zonų reikalavimai, ko negalima keisti"],
    ["FaBullseye", "Konkretus klausimas", "Vienas realus sprendimas: tinklo plėtra, zonavimas ar atkūrimas"]];
  for (let i = 0; i < needs.length; i++) {
    const col = i % 2, row = Math.floor(i / 2);
    const x = 0.6 + col * 4.05, y = 1.5 + row * 2.6, w = 3.8, h = 2.35;
    card(s, x, y, w, h, "need-card-" + i);
    await iconCircle(s, needs[i][0], x + 0.25, y + 0.3, 0.7, C.accent1, "need-" + i);
    txt(s, needs[i][1], { x: x + 1.15, y: y + 0.3, w: w - 1.35, h: 0.7, fontSize: 18, bold: true, fontFace: THEME.headFontFace, valign: "middle" });
    txt(s, needs[i][2], { x: x + 0.25, y: y + 1.15, w: w - 0.5, h: 1.05, fontSize: 14 });
  }
  card(s, 8.85, 1.5, 3.85, 4.95, "pilot-result", C.text2);
  await iconCircle(s, "FaHandshake", 9.15, 1.85, 0.8, C.accent3, "pilot");
  txt(s, "Rezultatas", { x: 9.15, y: 2.9, w: 3.25, h: 0.5, fontSize: 20, bold: true, color: C.background1, fontFace: THEME.headFontFace });
  txt(s, "Skaidrus, pakartojamas ir argumentuotas planas su alternatyvomis, kurį galima aptarti su savivaldybėmis, naudotojais ir bendruomene.", { x: 9.15, y: 3.45, w: 3.25, h: 2.6, fontSize: 16, color: C.background1 });
  s.addNotes("Bandomajam projektui reikėtų keturių dalykų: duomenų apie saugotinas vertybes, sąnaudų ar spaudimo sluoksnių, galiojančių taisyklių ir – svarbiausia – vieno realaus klausimo. Rezultatas – ne vienas „teisingas“ žemėlapis, o skaidrus ir pakartojamas planas su alternatyvomis, kurį galima aptarti su suinteresuotomis šalimis.");

  // ===== 16. Closing =====
  s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Toliau" });
  s.addText("Klausimai ir diskusija", { placeholder: "title" });
  s.addText("Kurie neišplėtoti funkcionalumai jums svarbiausi?\ngithub.com/razinkele/pymarxan · atvirasis kodas, MIT licencija", { placeholder: "body" });
  await iconCircle(s, "FaUsers", 0.8, 0.8, 0.9, C.accent1, "closing");
  s.addNotes("Ačiū. Labiausiai norėčiau išgirsti, kurie iš neišplėtotų dalykų jums būtų svarbiausi, ir ar yra teritorija, kurioje galėtume atlikti bandomąjį projektą.");

  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
})();
