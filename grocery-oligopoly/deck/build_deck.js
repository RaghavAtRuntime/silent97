// Builds deliverables/executive_presentation.pptx from outputs/tables/*.csv.
// Every number on a slide is read from the analysis tables at build time.
// Run: (cd deck && npm install && node build_deck.js)

const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");

const ROOT = path.resolve(__dirname, "..");
const TABLES = path.join(ROOT, "outputs", "tables");
const OUT = path.join(ROOT, "deliverables", "executive_presentation.pptx");

// ---------------------------------------------------------------- data
function parseCSV(text) {
  const rows = [];
  let row = [], field = "", q = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (q) {
      if (c === '"' && text[i + 1] === '"') { field += '"'; i++; }
      else if (c === '"') q = false;
      else field += c;
    } else if (c === '"') q = true;
    else if (c === ",") { row.push(field); field = ""; }
    else if (c === "\n") { row.push(field); rows.push(row); row = []; field = ""; }
    else if (c !== "\r") field += c;
  }
  if (field.length || row.length) { row.push(field); rows.push(row); }
  const [head, ...body] = rows.filter((r) => r.length > 1 || r[0] !== "");
  return body.map((r) => Object.fromEntries(head.map((h, i) => {
    const v = r[i];
    const n = Number(v);
    return [h, v !== "" && !Number.isNaN(n) && /^[-+0-9.eE]+$/.test(v) ? n : v];
  })));
}
const T = (name) => parseCSV(fs.readFileSync(path.join(TABLES, `${name}.csv`), "utf8"));
const pct = (x, d = 0) => `${(100 * x).toFixed(d)}%`;
const spct = (x, d = 1) => `${x >= 0 ? "+" : "−"}${Math.abs(100 * x).toFixed(d)}%`;
const find = (rows, f) => { const r = rows.find(f); if (!r) throw new Error("missing row"); return r; };

const cov = T("01_coverage");
const gaps = T("02_identical_item_gaps");
const gapCat = T("02b_gap_by_category_brand");
const ratio = T("02d_loblaws_nofrills_ratio_distribution")[0];
const idx = T("03b_banner_index_overall");
const sync = T("04_synchronisation");
const samePrice = T("05b_paired_change_same_new_price");
const weekly = T("06_weekly_change_rates");
const freezeSig = T("06c_freeze_window_significance");
const promos = T("07_promotions_by_banner");
const brand = T("08b_brand_architecture");
const infl = T("09d_inflation_summary");
const cpiCum = T("09c_statcan_cumulative");
const cpi = T("09b_statcan_cpi");
const kvi = T("10_kvi_entry_prices");
const weekday = T("11_change_weekday");
const cleaning = T("00_cleaning_log");

const gap = (a, b) => find(gaps, (r) => r.banner_a === a && r.banner_b === b);
const lift = (a, b, k = 1) => {
  const r1 = find(sync, (r) => r.banner_a === a && r.banner_b === b);
  const r2 = find(sync, (r) => r.banner_a === b && r.banner_b === a);
  return (r1[`lift_${k}d`] + r2[`lift_${k}d`]) / 2;
};
const ix = (b) => find(idx, (r) => r.banner === b);
const freezeRatio = (b) => {
  const g = weekly.filter((r) => r.banner === b);
  const inn = g.filter((r) => r.freeze_window === "True" || r.freeze_window === true);
  const out = g.filter((r) => !(r.freeze_window === "True" || r.freeze_window === true));
  const s = (a, k) => a.reduce((t, r) => t + r[k], 0);
  return (s(inn, "ups") / s(inn, "active")) / (s(out, "ups") / s(out, "active"));
};
const promoShare = (b) => find(cov, (r) => r.banner === b).promo_share;
const cum = (p) => find(cpiCum, (r) => r.product === p).cum_change_since_jan2020;
const totalRaw = find(cleaning, (r) => r.step.startsWith("1. raw price")).rows;
const totalDaily = find(cleaning, (r) => r.step.startsWith("4.")).rows;
const persistent = find(cleaning, (r) => r.step.startsWith("5a.")).rows;
const thursday = (b) => find(weekday, (r) => r.banner === b && r.weekday === "Thursday").share;

// ---------------------------------------------------------------- design
const C = {
  ink: "16202C", ink2: "4A5563", muted: "8A8F98", rule: "E3E6EA", bg: "FFFFFF", panel: "F3F5F7",
  navy: "14263D", navy2: "1F3A5F", accent: "E4572E",
  loblaws: "2A78D6", nofrills: "184F95", tt: "86B6EF", metro: "EB6834", voila: "1BAF7A", walmart: "6250D6",
  galleria: "E87BA4", saveon: "8A8984",
};
const BANNER = { "Loblaws": C.loblaws, "No Frills": C.nofrills, "T&T": C.tt, "Metro": C.metro, "Voilà": C.voila,
  "Walmart": C.walmart, "Galleria": C.galleria, "Save-On-Foods": C.saveon };
const HEAD = "Cambria", BODY = "Calibri";
const SRC_HAMMER = "Source: Project Hammer (J. Filipp) Toronto online shelf prices, Feb 2024–Oct 2026; author analysis.";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5 in
pres.title = "Same Owner, Different Shelf";
pres.subject = "Consumer-behaviour perspective on Canada's grocery oligopoly";
pres.author = "Independent research project";
pres.theme = { headFontFace: HEAD, bodyFontFace: BODY };

const W = 13.333, M = 0.6;

pres.defineSlideMaster({
  title: "TITLE_DARK",
  background: { color: C.navy },
  objects: [],
  slideNumber: undefined,
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: C.bg },
  margin: [0.5, 0.6, 0.6, 0.6],
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: M, y: 0.35, w: W - 2 * M, h: 1.05, fontFace: HEAD,
      fontSize: 28, bold: true, color: C.ink, valign: "top", align: "left", margin: 0 }, text: "" } },
    { text: { text: "Same Owner, Different Shelf · Canada grocery pricing", options: { x: M, y: 7.08, w: 7, h: 0.3,
      fontFace: BODY, fontSize: 9, color: C.muted, margin: 0 } } },
  ],
  slideNumber: { x: W - M - 0.5, y: 7.08, w: 0.5, h: 0.3, fontFace: BODY, fontSize: 9, color: C.muted, align: "right" },
});
pres.defineSlideMaster({
  title: "SECTION_DARK",
  background: { color: C.navy2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: M, y: 2.6, w: W - 2 * M, h: 1.2, fontFace: HEAD,
      fontSize: 40, bold: true, color: "FFFFFF", align: "left", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: M, y: 3.9, w: W - 2 * M, h: 1.0, fontFace: BODY,
      fontSize: 18, color: "C9D4E3", align: "left", margin: 0 }, text: "" } },
  ],
});

function src(slide, text) {
  slide.addText(text, { x: M, y: 6.72, w: W - 2 * M, h: 0.3, fontFace: BODY, fontSize: 10, color: C.muted,
    margin: 0, isTextBox: true, objectName: "Source" });
}
function chartBase(extra = {}) {
  return Object.assign({
    catAxisLabelColor: C.ink2, valAxisLabelColor: C.ink2, catAxisLabelFontFace: "+mn-lt",
    valAxisLabelFontFace: "+mn-lt", catAxisLabelFontSize: 12, valAxisLabelFontSize: 11,
    valGridLine: { color: C.rule, size: 0.75 }, catGridLine: { style: "none" },
    catAxisLineShow: false, valAxisLineShow: false, dataLabelFontFace: "+mn-lt", dataLabelFontSize: 11,
    dataLabelColor: C.ink, legendFontFace: "+mn-lt", legendFontSize: 11, legendColor: C.ink2,
    titleFontFace: "+mn-lt", titleFontSize: 13, titleColor: C.ink, showTitle: true,
  }, extra);
}
function takeaway(slide, x, y, w, h, heading, body, color = C.navy2) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: C.panel }, line: { color: C.panel },
    rectRadius: 0.08, objectName: "Takeaway panel" });
  slide.addText([
    { text: heading, options: { bold: true, color, fontSize: 15, breakLine: true } },
    { text: body, options: { color: C.ink2, fontSize: 14 } },
  ], { x: x + 0.25, y: y + 0.2, w: w - 0.5, h: h - 0.4, fontFace: BODY, valign: "top", margin: 0,
    paraSpaceAfter: 6, isTextBox: true, objectName: "Takeaway text" });
}
function stat(slide, x, y, w, value, label, color) {
  slide.addText(value, { x, y, w, h: 0.9, fontFace: HEAD, fontSize: 44, bold: true, color, margin: 0,
    isTextBox: true, objectName: "Stat value" });
  slide.addText(label, { x, y: y + 0.92, w, h: 1.1, fontFace: BODY, fontSize: 14, color: C.ink2, margin: 0,
    valign: "top", isTextBox: true, objectName: "Stat label" });
}
function badge(slide, x, y, n, color = C.accent) {
  slide.addShape(pres.shapes.OVAL, { x, y, w: 0.46, h: 0.46, fill: { color }, line: { color }, objectName: "Badge" });
  slide.addText(String(n), { x, y, w: 0.46, h: 0.46, align: "center", valign: "middle", fontFace: BODY,
    fontSize: 15, bold: true, color: "FFFFFF", margin: 0, isTextBox: true, objectName: "Badge number" });
}

// ---------------------------------------------------------------- slides
const LBNF = gap("Loblaws", "No Frills");
const LBME = gap("Loblaws", "Metro");
const LBWM = gap("Loblaws", "Walmart");
const MEWM = gap("Metro", "Walmart");
const NFWM = gap("No Frills", "Walmart");

// 1. Title
pres.addSection({ title: "Opening" });
let s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Opening" });
s.addText("CONSUMER BEHAVIOUR · PRICING STRATEGY · MARKET STRUCTURE", { x: M, y: 1.4, w: 11, h: 0.4, fontFace: BODY,
  fontSize: 14, bold: true, color: C.accent, charSpacing: 2, margin: 0, isTextBox: true });
s.addText("Same Owner, Different Shelf", { x: M, y: 1.95, w: 12, h: 1.2, fontFace: HEAD, fontSize: 54, bold: true,
  color: "FFFFFF", margin: 0, isTextBox: true });
s.addText("How Canada's grocery oligopoly prices a portfolio of banners — and what shoppers actually see",
  { x: M, y: 3.15, w: 11.5, h: 0.9, fontFace: BODY, fontSize: 22, color: "C9D4E3", margin: 0, isTextBox: true });
s.addText([
  { text: `${(totalRaw / 1e6).toFixed(1)}M shelf-price observations · 8 Toronto banners · Feb 2024–Oct 2026`,
    options: { breakLine: true } },
  { text: "Data: Project Hammer (banner prices) · Statistics Canada Tables 18-10-0004 and 18-10-0245" },
], { x: M, y: 5.4, w: 11.5, h: 0.9, fontFace: BODY, fontSize: 14, color: "9FB0C6", margin: 0, isTextBox: true });
s.addNotes("This deck summarises an independent research project on how Canada's large grocery parents price across "
  + "their banners. It uses daily online shelf prices for eight Toronto banners from Project Hammer, matched product by "
  + "product, plus Statistics Canada CPI and average retail prices for context. The question: do these firms behave like "
  + "independent merchants competing item by item, or like portfolio managers who segment shoppers by value proposition?");

// 2. Executive summary
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Opening" });
s.addText("The market behaves like managed price tiers, not a free-for-all", { placeholder: "title" });
const colW = (W - 2 * M - 0.9) / 4;
const stats = [
  [spct(LBNF.median_regular_premium_a_vs_b), "Premium for the identical item under the Loblaws banner vs No Frills — same parent, same day",
    C.loblaws],
  [pct(LBME.share_days_identical_regular), "of matched item-days where rival conventional banners Loblaws and Metro charge the identical regular price",
    C.metro],
  [`${lift("Loblaws", "No Frills").toFixed(1)}×`,
    `Price-change synchronisation inside the Loblaw portfolio, vs ${lift("Loblaws", "Metro").toFixed(1)}× for Loblaws–Metro (1.0× = chance)`, C.navy2],
  [`${freezeRatio("Loblaws").toFixed(2)}`, "Loblaws' increase rate inside vs outside the Nov–Feb “price freeze” window: no visible freeze",
    C.accent],
];
stats.forEach(([v, l, c], i) => stat(s, M + i * (colW + 0.3), 1.65, colW, v, l, c));
takeaway(s, M, 4.35, W - 2 * M, 1.95, "So what",
  "Shoppers face a ladder of price tiers — conventional banners at one price point, discount banners at another — "
  + "and the largest parent owns a rung on both. Competition shows up between tiers more than within them, and a shopper "
  + "who trades down to save money often stays inside the same corporate portfolio. Price perception is shaped as much by "
  + "banner architecture and “was” prices as by the price of any single item.");
src(s, SRC_HAMMER);
s.addNotes("Four numbers carry the story. First, inside Loblaw the identical SKU is about 12% dearer under the Loblaws banner "
  + "than at No Frills: the banner itself is a price signal. Second, across owners, the two conventional banners, Loblaws and "
  + "Metro, sit on the identical regular price on roughly 40% of matched item-days, whereas Loblaws and Walmart almost never do. "
  + "Third, timing coordination is visible inside the Loblaw portfolio but across owners it is no higher than chance. Fourth, "
  + "we tested a publicly described Nov–Feb price freeze and could not detect it in the data.");

// 3. Why it matters — StatCan context
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Opening" });
s.addText(`Grocery prices are up ${pct(cum("Food purchased from stores"))} since 2020, outpacing overall inflation`,
  { placeholder: "title" });
const cpiCan = (prod) => cpi.filter((r) => r.geo === "Canada" && r.product === prod && r.month >= "2019-01-01")
  .sort((a, b) => (a.month < b.month ? -1 : 1));
const base = (rows) => rows.find((r) => r.month.startsWith("2020-01")).value;
const food = cpiCan("Food purchased from stores"), all = cpiCan("All-items");
const labels = food.map((r) => r.month.slice(0, 7));
s.addChart(pres.charts.LINE, [
  { name: "Food purchased from stores", labels, values: food.map((r) => +(100 * r.value / base(food)).toFixed(1)) },
  { name: "All items", labels, values: all.map((r) => +(100 * r.value / base(all)).toFixed(1)) },
], chartBase({ x: M, y: 1.55, w: 7.6, h: 5.0, chartColors: [C.accent, C.muted], lineSize: 2.5, lineDataSymbol: "none",
  title: "Consumer Price Index, Canada (Jan 2020 = 100)", showLegend: true, legendPos: "b", valAxisMinVal: 90,
  catAxisLabelFrequency: 12, catAxisLabelRotate: 0 }));
takeaway(s, 8.6, 1.55, W - M - 8.6, 2.45, "A concentrated market",
  "The Competition Bureau's 2023 market study counted five large chains (Loblaw, Sobeys, Metro, Costco, Walmart) and "
  + "found grocery gross margins up modestly — roughly 1–2 points since 2017.");
takeaway(s, 8.6, 4.15, W - M - 8.6, 2.35, "The behavioural question",
  "When a few parents run many banners, does a “cheaper store” signal real competition — or a different rung of the same ladder?",
  C.accent);
src(s, "Sources: Statistics Canada Table 18-10-0004-01 (CPI, NSA); Competition Bureau, Retail Grocery Market Study (June 2023).");
s.addNotes(`Context: StatCan's CPI for food purchased from stores is up ${pct(cum("Food purchased from stores"))} since January 2020, versus `
  + `${pct(cum("All-items"))} for all items. The Competition Bureau's 2023 market study described an industry concentrated in five `
  + "large chains with modestly rising margins. That combination is what makes the consumer question sharp: in a concentrated "
  + "market, store choice may not be the competitive check shoppers think it is.");

// 4. Portfolio map
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Opening" });
s.addText("Each parent spans several value propositions under different banner names", { placeholder: "title" });
const tiers = ["Discount", "Conventional", "Specialty / premium", "Online"];
const parents = [
  ["Loblaw", C.loblaws, ["No Frills*, Real Canadian Superstore, Maxi", "Loblaws*, Zehrs, Fortinos, Provigo", "T&T*", "PC Express"]],
  ["Empire (Sobeys)", C.voila, ["FreshCo", "Sobeys, Safeway, IGA, Foodland", "Farm Boy, Longo's", "Voilà*"]],
  ["Metro Inc.", C.metro, ["Food Basics, Super C", "Metro*", "Adonis", "metro.ca"]],
  ["Walmart Canada", C.walmart, ["Walmart* (EDLP)", "—", "—", "walmart.ca"]],
];
const gx = M + 2.3, gw = (W - M - gx) / 4, gy = 1.55, gh = 0.98;
tiers.forEach((t, j) => s.addText(t, { x: gx + j * gw, y: gy, w: gw - 0.1, h: 0.45, fontFace: BODY, fontSize: 14,
  bold: true, color: C.ink2, align: "center", margin: 0, isTextBox: true }));
parents.forEach(([p, col, cells], i) => {
  const y = gy + 0.55 + i * (gh + 0.08);
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: M, y, w: 2.15, h: gh, fill: { color: col }, line: { color: col },
    rectRadius: 0.08, objectName: `Parent ${p}` });
  s.addText(p, { x: M, y, w: 2.15, h: gh, fontFace: BODY, fontSize: 15, bold: true, color: "FFFFFF", align: "center",
    valign: "middle", margin: 0.05, isTextBox: true });
  cells.forEach((c, j) => {
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: gx + j * gw, y, w: gw - 0.1, h: gh,
      fill: { color: c === "—" ? "FFFFFF" : C.panel }, line: { color: C.rule, width: 0.75 }, rectRadius: 0.06 });
    s.addText(c, { x: gx + j * gw + 0.08, y, w: gw - 0.26, h: gh, fontFace: BODY, fontSize: 12, color: C.ink,
      align: "center", valign: "middle", margin: 0, isTextBox: true });
  });
});
src(s, "* = banner observed in the price data. Ownership from corporate disclosures (Loblaw acquired T&T in 2009). "
  + "Banner lists are illustrative, not exhaustive. Galleria (independent) and Save-On-Foods (Pattison, Western Canada) also in data.");
s.addNotes("The structural fact behind the analysis: each large parent runs a discount banner, a conventional banner and often "
  + "specialty or online formats. The asterisks mark banners in our price data. Loblaw is unusually well covered: we observe its "
  + "conventional banner, its discount banner and T&T, which share a product catalogue and so can be compared SKU for SKU.");

// 5. Data & method
pres.addSection({ title: "Approach" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Approach" });
s.addText("Daily shelf prices, matched product by product across banners", { placeholder: "title" });
const steps = [
  ["Collect", `${(totalRaw / 1e6).toFixed(1)}M daily price records scraped from 8 banners' websites (Toronto pickup prices), plus StatCan CPI and average retail prices.`],
  ["Clean (Python · DuckDB)", "Parse multi-buy and deal prices, repair cents-scaled values, separate regular from promotional prices, "
    + `collapse to ${(totalDaily / 1e6).toFixed(1)}M banner-product-days.`],
  ["Match", "Identical items via shared SKU (Loblaws ↔ No Frills) or UPC across owners; variable-weight items excluded."],
  ["Measure", `Identical-item gaps, basket price index, ${Math.round(persistent / 1000)}k persistent regular-price changes `
    + "tested for synchronisation against a time-shifted placebo, promotion and freeze-window tests."],
  ["Translate (Excel · deck)", "Workbook with live formulas and charts; memo and deck framed around shopper decision-making."],
];
steps.forEach(([h, b], i) => {
  const y = 1.6 + i * 0.98;
  badge(s, M, y + 0.05, i + 1, C.navy2);
  s.addText([{ text: h, options: { bold: true, color: C.ink, fontSize: 15, breakLine: true } },
    { text: b, options: { color: C.ink2, fontSize: 14 } }],
  { x: M + 0.65, y, w: 6.9, h: 0.95, fontFace: BODY, valign: "top", margin: 0, isTextBox: true });
});
const inflBanners = ["Voilà", "Metro", "Loblaws", "No Frills", "Walmart"];
const inflRow = (b) => find(infl, (r) => r.banner === b).change_since_aug2024;
const cpiOn = find(infl, (r) => r.banner === "CPI food from stores — Ontario").change_since_aug2024;
s.addChart(pres.charts.BAR, [{ name: "Change since Aug 2024", labels: [...inflBanners, "StatCan CPI (ON)"],
  values: [...inflBanners.map(inflRow), cpiOn].map((v) => +(100 * v).toFixed(1)) }],
chartBase({ x: 8.0, y: 1.55, w: W - M - 8.0, h: 4.95, barDir: "bar", chartColors: [...inflBanners.map((b) => BANNER[b]), C.ink],
  showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: '0.0"%"', showLegend: false, valAxisHidden: true,
  valGridLine: { style: "none" }, title: "Validation: price change Aug 2024 → latest (%)", catAxisOrientation: "maxMin" }));
src(s, "Sources: Project Hammer; Statistics Canada Tables 18-10-0004-01 and 18-10-0245-01. Banner index = chained matched-model "
  + "index of regular prices; CPI to Aug 2026.");
s.addNotes("Method in one line: clean the scrape, separate the regular shelf price from promotions, match identical items, "
  + "then measure levels, timing and promotions. The validation chart shows the scraped banners' regular-price inflation lands "
  + "in the same range as StatCan's CPI for food purchased from stores, which gives confidence the data reflect real shelf prices. "
  + "Synchronised or identical prices are descriptive evidence only; they are also consistent with common supplier costs.");

// 6. Price ladder
pres.addSection({ title: "Findings" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Findings" });
s.addText("One basket, two price tiers — and Loblaw owns a rung in each", { placeholder: "title" });
const core = ["Metro", "Loblaws", "No Frills", "Walmart"];
s.addChart(pres.charts.BAR, [
  { name: "Regular price", labels: core, values: core.map((b) => +ix(b).regular_index.toFixed(1)) },
  { name: "After promotions", labels: core, values: core.map((b) => +ix(b).shelf_vs_regular_base.toFixed(1)) },
], chartBase({ x: M, y: 1.55, w: 7.8, h: 5.0, barDir: "col", barGrouping: "clustered", chartColors: [C.navy2, "9FB0C6"],
  showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0", valAxisMinVal: 70, valAxisMaxVal: 120,
  showLegend: true, legendPos: "b", title: "Identical-basket price index (4-banner regular average = 100)", barGapWidthPct: 60 }));
takeaway(s, 8.75, 1.55, W - M - 8.75, 2.6, "Tiering, not head-to-head",
  `Metro (${ix("Metro").regular_index.toFixed(0)}) and Loblaws (${ix("Loblaws").regular_index.toFixed(0)}) sit together; `
  + `No Frills (${ix("No Frills").regular_index.toFixed(0)}) and Walmart (${ix("Walmart").regular_index.toFixed(0)}) sit together, `
  + `about ${(ix("Loblaws").regular_index - ix("No Frills").regular_index).toFixed(0)} points lower.`);
takeaway(s, 8.75, 4.3, W - M - 8.75, 2.2, "Behavioural lens",
  "The conventional tier anchors the shopper's sense of the “normal” price; the discount tier then reads as a bargain — "
  + "even when both belong to one parent.", C.accent);
src(s, SRC_HAMMER + ` ~${Math.round(ix("Metro").avg_basket_items).toLocaleString()} UPCs stocked by all four banners each month, Aug 2024+.`);
s.addNotes("On an identical basket of roughly 1,700 UPCs carried by all four banners, prices sort into two tiers. Metro and "
  + "Loblaws are about 10-13% above the four-banner average; No Frills and Walmart about 10% below. Promotions bring the conventional "
  + "banners down to roughly the average but do not close the gap with the discount tier. The structurally interesting point "
  + "is that Loblaw occupies both tiers.");

// 7. Banner premium inside Loblaw
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Findings" });
s.addText(`Inside one portfolio, the same SKU costs ${pct(LBNF.median_regular_premium_a_vs_b)} more under the Loblaws banner`,
  { placeholder: "title" });
const cats = gapCat.filter((r) => r.banner_a === "Loblaws" && r.banner_b === "No Frills" && r.category !== "Other"
  && r.category !== "Household & personal care");
const catNames = [...new Set(cats.map((r) => r.category))];
const catVal = (c, pl) => { const r = cats.find((x) => x.category === c && String(x.private_label) === pl); return r ? +(100 * r.median_regular_premium).toFixed(1) : 0; };
catNames.sort((a, b) => catVal(b, "False") - catVal(a, "False"));
s.addChart(pres.charts.BAR, [
  { name: "National brands", labels: catNames, values: catNames.map((c) => catVal(c, "False")) },
  { name: "Loblaw private label (PC, No Name)", labels: catNames, values: catNames.map((c) => catVal(c, "True")) },
], chartBase({ x: M, y: 1.55, w: 7.8, h: 5.0, barDir: "bar", barGrouping: "clustered", chartColors: [C.loblaws, C.tt],
  showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: '0"%"', showLegend: true, legendPos: "b",
  valAxisHidden: true, valGridLine: { style: "none" }, catAxisOrientation: "maxMin",
  title: "Median Loblaws premium over No Frills, identical SKU (%)", barGapWidthPct: 50 }));
stat(s, 8.75, 1.6, 4.0, pct(LBNF.share_items_a_dearer), `of ${LBNF.matched_products.toLocaleString()} matched SKUs are dearer at Loblaws; only ${pct(LBNF.share_items_a_cheaper)} cheaper`, C.loblaws);
takeaway(s, 8.75, 3.7, W - M - 8.75, 2.8, "The banner is the price signal",
  `Half of SKUs carry a premium between ${spct(ratio.p25 - 1, 0)} and ${spct(ratio.p75 - 1, 0)}; ${pct(ratio.share_over_20pct)} exceed +20%. `
  + "The premium holds for Loblaw's own brands too — the shopper pays for the store environment, not the product.");
src(s, SRC_HAMMER + " Regular (non-promotional) prices; distribution from Jul–Sep 2026.");
s.addNotes("This is the cleanest test of portfolio pricing because both banners share one catalogue. The identical SKU is a "
  + "median 12-13% dearer at Loblaws; about four in five SKUs are dearer there, and the gap is present in every category, including "
  + "President's Choice and No Name. For a shopper, trading down from Loblaws to No Frills is a real saving - but the revenue stays "
  + "with the same parent. That is the portfolio logic: margin from the conventional banner, volume defence from the discount banner.");

// 8. Conventional rivals share price points
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Findings" });
s.addText("Rival conventional banners land on identical price points; tiers rarely do", { placeholder: "title" });
const pairs = [["Loblaws", "Metro"], ["Metro", "Voilà"], ["Loblaws", "Voilà"], ["Loblaws", "No Frills"],
  ["No Frills", "Metro"], ["No Frills", "Walmart"], ["Loblaws", "Walmart"], ["Metro", "Walmart"]];
s.addChart(pres.charts.BAR, [{ name: "Identical regular price", labels: pairs.map(([a, b]) => `${a} – ${b}`),
  values: pairs.map(([a, b]) => +(100 * gap(a, b).share_days_identical_regular).toFixed(1)) }],
chartBase({ x: M, y: 1.55, w: 7.8, h: 5.0, barDir: "bar", catAxisOrientation: "maxMin",
  chartColors: pairs.map(([a, b]) => (a === "Loblaws" && b === "No Frills" ? C.loblaws : C.metro)),
  showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: '0"%"', showLegend: false, valAxisHidden: true,
  valGridLine: { style: "none" }, title: "Share of matched item-days at the identical regular price (%)", barGapWidthPct: 45 }));
const sp = (a, b) => find(samePrice, (r) => r.banner_a === a && r.banner_b === b).share_new_price_identical;
stat(s, 8.75, 1.6, 4.0, pct(sp("Loblaws", "Metro")), "of paired Loblaws–Metro price changes land on the identical new price "
  + `(Loblaws–Walmart: ${pct(sp("Loblaws", "Walmart"))})`, C.metro);
takeaway(s, 8.75, 3.7, W - M - 8.75, 2.8, "Read with care",
  "Shared price points are consistent with tacit coordination, but also with manufacturer-suggested retail prices and "
  + "common list-price changes. Either way, the shopper comparing conventional banners finds little to gain.");
src(s, SRC_HAMMER + " Same-parent pair shown in blue.");
s.addNotes(`Across owners, Loblaws and Metro charge the identical regular price on ${pct(LBME.share_days_identical_regular)} of matched `
  + `item-days and have a median gap of zero. Against Walmart, the same Loblaws items are a median ${spct(LBWM.median_regular_premium_a_vs_b, 0)} dearer and `
  + "almost never identical. When Loblaws and Metro both change an item's price within two weeks, about 29% of the time they land on "
  + "the exact same new price. We are careful: this is consistent with suggested retail prices from suppliers, not proof of collusion.");

// 9. Timing / synchronisation
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Findings" });
s.addText("Price changes cluster inside the Loblaw portfolio, far less across owners", { placeholder: "title" });
const sPairs = [
  ["Loblaws", "No Frills", "Same parent", C.loblaws], ["Metro", "Walmart", "Rival owners", C.metro],
  ["No Frills", "Walmart", "Rival owners", C.metro], ["No Frills", "Metro", "Rival owners", C.metro],
  ["Loblaws", "Walmart", "Rival owners", C.metro], ["Loblaws", "Metro", "Rival owners", C.metro],
  ["Metro", "Save-On-Foods", "Out-of-market control", C.saveon], ["No Frills", "Save-On-Foods", "Out-of-market control", C.saveon],
  ["Loblaws", "Save-On-Foods", "Out-of-market control", C.saveon],
];
s.addChart(pres.charts.BAR, [{ name: "Lift", labels: sPairs.map(([a, b]) => `${a} – ${b}`),
  values: sPairs.map(([a, b]) => +lift(a, b).toFixed(2)) }],
chartBase({ x: M, y: 1.55, w: 7.8, h: 5.0, barDir: "bar", catAxisOrientation: "maxMin", chartColors: sPairs.map((p) => p[3]),
  showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: '0.00"×"', showLegend: false, valAxisMinVal: 0,
  valAxisMaxVal: 2.5, title: "Synchronisation lift, ±1 day (1.0 = chance; blue same parent, orange rivals, grey control)",
  barGapWidthPct: 45, valAxisLabelFormatCode: '0.0"×"' }));
takeaway(s, 8.75, 1.55, W - M - 8.75, 2.55, "Coordination lives inside portfolios",
  `Loblaws and No Frills change identical SKUs together ${lift("Loblaws", "No Frills").toFixed(1)}× as often as chance. `
  + "Rival owners sit near 1× — in the same range as Save-On-Foods, a Western grocer outside the Toronto market.");
takeaway(s, 8.75, 4.25, W - M - 8.75, 2.25, "A weekly rhythm",
  `${pct(thursday("Loblaws"))} of Loblaws' and ${pct(thursday("Metro"))} of Metro's regular-price changes appear on a Thursday, `
  + "the flyer cycle. No banner consistently leads.", C.accent);
src(s, SRC_HAMMER + " Placebo shifts each change date ±2–6 months; lift averaged over both directions of the pair.");
s.addNotes("We asked whether regular-price changes on identical items happen at the same time, relative to a placebo where the "
  + "dates are shifted by months. Inside Loblaw, synchronisation is about twice chance. Across rival owners, it ranges from slightly "
  + "below to about 1.4 times chance - the same range we see with Save-On-Foods, which does not operate in Toronto and therefore "
  + "only shares supplier cost shocks. So the data do not show rivals reacting to each other day by day; the structure is maintained "
  + "through tiers and price points, not rapid matching. Lead-lag analysis shows no consistent price leader.");

// 10. Promotions
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Findings" });
s.addText("Conventional banners run high-low promotions that narrow — but never close — the gap", { placeholder: "title" });
const pb = ["Metro", "Loblaws", "Walmart", "Voilà", "No Frills"];
s.addChart(pres.charts.BAR, [{ name: "Promo share", labels: pb, values: pb.map((b) => +(100 * promoShare(b)).toFixed(1)) }],
  chartBase({ x: M, y: 1.55, w: 6.0, h: 5.0, barDir: "col", chartColors: pb.map((b) => BANNER[b]), showValue: true,
    dataLabelPosition: "outEnd", dataLabelFormatCode: '0"%"', showLegend: false, valAxisHidden: true,
    valGridLine: { style: "none" }, title: "Share of product-days on promotion (%)", barGapWidthPct: 45 }));
s.addChart(pres.charts.BAR, [{ name: "Realised promo discount", labels: core,
  values: core.map((b) => +(100 * ix(b).realised_promo_discount).toFixed(1)) }],
chartBase({ x: 6.85, y: 1.55, w: 3.4, h: 5.0, barDir: "col", chartColors: core.map((b) => BANNER[b]), showValue: true,
  dataLabelPosition: "outEnd", dataLabelFormatCode: '0.0"%"', showLegend: false, valAxisHidden: true,
  valGridLine: { style: "none" }, title: "Avg basket saving from promos (%)", barGapWidthPct: 45, catAxisLabelFontSize: 11 }));
takeaway(s, 10.45, 1.55, W - M - 10.45, 4.95, "Reference prices at work",
  "A struck-through “was” price frames the deal against the conventional tier's high regular price. "
  + "Discount banners promote less: their everyday price is the message.");
src(s, SRC_HAMMER + " Promotion = 'was' price, multi-buy or sale flag. Basket saving on the identical-basket index.");
s.addNotes(`Metro has about ${pct(promoShare("Metro"))} of product-days on promotion and Loblaws ${pct(promoShare("Loblaws"))}, versus `
  + `${pct(promoShare("No Frills"))} at No Frills. On the identical basket, promotions save shoppers roughly 9-10% at the conventional `
  + "banners and 4-7% at the discount banners. Behaviourally, promotions work through reference prices: the higher the regular "
  + "price, the bigger the visible 'saving', which keeps the conventional banners' price image competitive without moving the regular price.");

// 11. Freeze test
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Findings" });
s.addText("Testing a claim: no measurable Nov–Feb “price freeze” at Toronto banners", { placeholder: "title" });
const fb = ["Loblaws", "No Frills", "Metro", "Walmart", "Voilà"];
s.addChart(pres.charts.BAR, [{ name: "Ratio", labels: fb, values: fb.map((b) => +freezeRatio(b).toFixed(2)) }],
  chartBase({ x: M, y: 1.55, w: 7.8, h: 5.0, barDir: "col", chartColors: fb.map((b) => BANNER[b]), showValue: true,
    dataLabelPosition: "outEnd", dataLabelFormatCode: '0.00"×"', showLegend: false, valAxisMinVal: 0, valAxisMaxVal: 2,
    title: "Rate of regular-price increases: Nov 1–Feb 5 window vs rest of year (1.0 = no difference)", barGapWidthPct: 50,
    valAxisLabelFormatCode: '0.0"×"' }));
const p = (b) => find(freezeSig, (r) => r.banner === b).mann_whitney_p;
takeaway(s, 8.75, 1.55, W - M - 8.75, 2.6, "What we tested",
  "Metro publicly described a Nov 1–Feb 5 freeze on regular prices as an “industry practice”. If true, increases should "
  + "drop sharply in that window.");
takeaway(s, 8.75, 4.3, W - M - 8.75, 2.2, "What we found",
  `Increase rates in the window are ${freezeRatio("No Frills").toFixed(2)}–${freezeRatio("Metro").toFixed(2)}× the rest of the year `
  + `for the big Toronto banners (Mann-Whitney p ≥ ${Math.min(p("Loblaws"), p("No Frills"), p("Metro"), p("Walmart")).toFixed(2)}). `
  + "Any freeze is not visible on the shelf.", C.accent);
src(s, SRC_HAMMER + " Two windows (2024-25, 2025-26); persistent increases per 1,000 active products. Metro statement as quoted on the Project Hammer site.");
s.addNotes("A good research habit is to test public claims. Metro has said a November-to-February freeze is an industry practice. "
  + "If so, regular-price increases should drop in that window. They do not, in any meaningful way, at Loblaws, No Frills, Metro or "
  + "Walmart, across two winters. A freeze may apply to supplier cost increases rather than shelf prices, or to specific items; "
  + "but shoppers should not assume holiday prices are protected.");

// 12. Brand architecture
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Findings" });
s.addText("Store brands steer the shelf: up to one in five products at the big three's banners", { placeholder: "title" });
const bb = ["Voilà", "No Frills", "Loblaws", "Metro", "T&T", "Walmart"];
const br = (b) => find(brand, (r) => r.banner === b);
s.addChart(pres.charts.BAR, [
  { name: "Share of assortment", labels: bb, values: bb.map((b) => +(100 * br(b).pl_share_assortment).toFixed(1)) },
  { name: "Share of promotion-days", labels: bb, values: bb.map((b) => +(100 * br(b).pl_share_promo_days).toFixed(1)) },
], chartBase({ x: M, y: 1.55, w: 7.8, h: 5.0, barDir: "col", barGrouping: "clustered", chartColors: [C.navy2, C.accent],
  showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: '0"%"', showLegend: true, legendPos: "b",
  valAxisHidden: true, valGridLine: { style: "none" }, title: "Private-label share by banner (%)", barGapWidthPct: 50 }));
takeaway(s, 8.75, 1.55, W - M - 8.75, 2.6, "Architecture beyond the banner",
  "PC and No Name appear across Loblaw's banners at different prices, giving one brand a good-better-best ladder "
  + "inside a single store and across stores.");
takeaway(s, 8.75, 4.3, W - M - 8.75, 2.2, "Default effects",
  "Private label at eye level and in promotions shapes which option shoppers see first — a quiet lever on basket mix.", C.accent);
src(s, SRC_HAMMER + " Private label identified by brand/name (PC, No Name, Selection, Irresistibles, Compliments, Great Value…).");
s.addNotes(`Private label is ${pct(br("Loblaws").pl_share_assortment)} of the Loblaws assortment and ${pct(br("Voilà").pl_share_assortment)} at Voilà, `
  + `versus ${pct(br("Walmart").pl_share_assortment)} at Walmart. At Loblaws, store brands take a larger share of promotion-days than of the shelf, `
  + "meaning promotions tilt shoppers toward the retailer's own margin-rich products.");

// 13. Consumer behaviour interpretation
pres.addSection({ title: "Implications" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Implications" });
s.addText("Four behavioural mechanisms turn portfolio structure into market power", { placeholder: "title" });
const mech = [
  ["Choice-set illusion", "Many banner names signal many competitors. In our Toronto data, 5 of 8 banners belong to the big three parents, "
    + "and switching between Loblaws and No Frills keeps spending in one portfolio."],
  ["Anchoring & reference prices", "The conventional tier's regular price sets the internal reference; “was” prices and the discount "
    + "tier are judged against it, so both feel like savings."],
  ["Tier sorting (compromise effect)", "Shoppers self-select into good / better / best banners by value proposition rather "
    + `than comparing identical items — the ${pct(LBNF.median_regular_premium_a_vs_b)} banner premium is paid for environment and convenience.`],
  ["Price image over price facts", "Few shoppers can compare 1,700 items; they rely on a banner's price image. Identical price points "
    + "among conventional rivals blunt comparison shopping where it would matter most."],
];
mech.forEach(([h, b], i) => {
  const x = M + (i % 2) * ((W - 2 * M) / 2 + 0.15), y = 1.6 + Math.floor(i / 2) * 2.5, w = (W - 2 * M) / 2 - 0.15;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h: 2.3, fill: { color: C.panel }, line: { color: C.panel },
    rectRadius: 0.08 });
  badge(s, x + 0.25, y + 0.25, i + 1);
  s.addText([{ text: h, options: { bold: true, fontSize: 16, color: C.ink, breakLine: true } },
    { text: b, options: { fontSize: 15, color: C.ink2 } }],
  { x: x + 0.9, y: y + 0.2, w: w - 1.15, h: 1.95, fontFace: BODY, valign: "top", margin: 0, paraSpaceAfter: 6, isTextBox: true });
});
src(s, "Interpretation of the price evidence through standard consumer-behaviour constructs; no shopper survey was conducted.");
s.addNotes("This is where the consumer-behaviour lens adds to the economics. The same price data look different once we remember "
  + "that shoppers do not compare 1,700 items; they choose stores using heuristics. Banner proliferation creates an illusion of choice; "
  + "the conventional tier anchors reference prices; shoppers sort themselves into tiers; and price image substitutes for price facts. "
  + "None of these require explicit coordination to sustain margins - the architecture does the work.");

// 14. Implications
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Implications" });
s.addText("What each stakeholder should do with this", { placeholder: "title" });
const imps = [
  ["Policymakers & regulators", ["Assess concentration at parent level, not banner level", "Mandate comparable unit pricing and "
    + "‘was’-price rules", "Use open scrape data (like Project Hammer) for ongoing monitoring"], C.navy2],
  ["Challengers & independents", ["Position against the conventional tier, where identical price points leave room",
    "Win on transparent everyday pricing rather than high-low", "Target categories with the widest banner premium"], C.voila],
  ["National-brand manufacturers", ["Recognise suggested prices become shared focal points", "Balance trade spend across tiers, "
    + "not just banners", "Watch private-label promotion share"], C.metro],
  ["Shoppers", ["Compare across owners, not just banners", "Treat ‘was’ prices as anchors, not savings", "The same SKU costs a median "
    + `${pct(LBNF.median_regular_premium_a_vs_b)} more at Loblaws than at No Frills`], C.accent],
];
const iw = (W - 2 * M - 0.6) / 4;
imps.forEach(([h, items, col], i) => {
  const x = M + i * (iw + 0.2);
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.6, w: iw, h: 4.9, fill: { color: C.panel }, line: { color: C.panel },
    rectRadius: 0.08 });
  s.addShape(pres.shapes.OVAL, { x: x + 0.25, y: 1.85, w: 0.4, h: 0.4, fill: { color: col }, line: { color: col } });
  s.addText(h, { x: x + 0.25, y: 2.35, w: iw - 0.5, h: 0.8, fontFace: BODY, fontSize: 16, bold: true, color: C.ink,
    valign: "top", margin: 0, isTextBox: true });
  s.addText(items.map((t, k) => ({ text: t, options: { bullet: true, breakLine: k < items.length - 1 } })),
    { x: x + 0.25, y: 3.2, w: iw - 0.45, h: 3.1, fontFace: BODY, fontSize: 14, color: C.ink2, valign: "top", margin: 0,
      paraSpaceAfter: 8, isTextBox: true });
});
src(s, "Recommendations are the author's interpretation of the evidence in this deck.");
s.addNotes("Implications by audience. For regulators, the unit of analysis should be the parent, because the banner is a marketing "
  + "device. For challengers, the opportunity is the conventional tier, where rivals share price points. For manufacturers, suggested "
  + "prices become shared focal points. For shoppers, the practical advice is to compare across owners and to treat 'was' prices as anchors.");

// 15. Limits & next steps
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Implications" });
s.addText("Limits of the evidence and how to extend it", { placeholder: "title" });
const lim = ["Online pickup prices for one Toronto neighbourhood — not every store, region or in-store price",
  "No cost or volume data: margins and market shares are not observed",
  "Synchronised or identical prices ≠ collusion; common supplier costs and suggested prices also explain them",
  "UPC matching for Loblaw banners was fuzzy in the source; sanity checks applied, some error remains",
  "Food Basics, FreshCo, Sobeys stores and Costco are not in the scrape"];
const nxt = ["Add a shopper survey: awareness of banner ownership, perceived price image, trade-down paths",
  "Link to supplier list-price changes to separate cost pass-through from strategic moves",
  "Extend to the Empire and Metro discount banners as data allow",
  "Track the Grocery Code of Conduct and unit-pricing changes as natural experiments"];
const listBox = (x, title, items, col) => {
  s.addText(title, { x, y: 1.6, w: 5.8, h: 0.5, fontFace: BODY, fontSize: 18, bold: true, color: col, margin: 0, isTextBox: true });
  s.addText(items.map((t, k) => ({ text: t, options: { bullet: true, breakLine: k < items.length - 1 } })),
    { x, y: 2.2, w: 5.8, h: 4.3, fontFace: BODY, fontSize: 15, color: C.ink2, valign: "top", margin: 0, paraSpaceAfter: 10,
      isTextBox: true });
};
listBox(M, "Limits", lim, C.ink);
listBox(M + 6.3, "Next steps", nxt, C.accent);
src(s, "See README and memo for full methodology and data definitions.");
s.addNotes("Be explicit about limits: these are online prices for one neighbourhood, there are no costs or volumes, and the timing "
  + "and price-point evidence is descriptive. The most valuable extension is a shopper survey to measure awareness of banner ownership "
  + "and perceived price image directly, which would test the behavioural mechanisms rather than infer them.");

// 16. Closing
s = pres.addSlide({ masterName: "SECTION_DARK", sectionTitle: "Implications" });
s.addText("The shelf is a portfolio, not a marketplace", { placeholder: "title" });
s.addText("Premium banners protect margin, discount banners defend volume, and rival conventional banners share price points — "
  + "so the shopper's real choice is a tier, not a competitor.", { placeholder: "body" });
s.addNotes("Close on the single idea: in Canadian grocery, the decision shoppers think they are making - which store - is largely a "
  + "choice of tier within a small number of portfolios.");

// 17. Appendix: sources
pres.addSection({ title: "Appendix" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Appendix" });
s.addText("Sources and definitions", { placeholder: "title" });
const refs = [
  "Filipp, J. Project Hammer: Canadian grocery price data (2024–2026). jacobfilipp.com/hammer · projecthammer.org",
  "Statistics Canada. Table 18-10-0004-01, Consumer Price Index, monthly, not seasonally adjusted. doi.org/10.25318/1810000401-eng",
  "Statistics Canada. Table 18-10-0245-01, Monthly average retail prices for selected products. doi.org/10.25318/1810024501-eng",
  "Competition Bureau Canada (2023). Canada needs more grocery competition: Retail Grocery Market Study Report.",
  "Nakamura, E. & Steinsson, J. (2008). Five facts about prices. Quarterly Journal of Economics — basis for the temporary-sale filter.",
  "Regular price: shelf price without promotion (or the 'was' price when promoted). Persistent change: not reversed within 14 days.",
  "Synchronisation lift: matched-change rate ÷ placebo rate with dates shifted ±2–6 months. Index: geometric mean on identical UPC basket.",
];
s.addText(refs.map((t, k) => ({ text: t, options: { bullet: true, breakLine: k < refs.length - 1 } })),
  { x: M, y: 1.6, w: W - 2 * M, h: 4.9, fontFace: BODY, fontSize: 14, color: C.ink2, valign: "top", margin: 0,
    paraSpaceAfter: 10, isTextBox: true });
s.addNotes("Full source list and the key definitions used throughout.");

pres.writeFile({ fileName: OUT }).then(() => console.log(`wrote ${path.relative(ROOT, OUT)}`));
