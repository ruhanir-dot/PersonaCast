(() => {
  const TARGET_DATE = "2026-09-15";
  const lines = document.body.innerText
    .split("\n")
    .map(x => x.trim())
    .filter(Boolean);

  const results = [];

  for (let i = 0; i < lines.length; i++) {
    const match = lines[i].match(
      /^搜尋\s*[「"](.+?)[」"]$/i
    );

    if (!match) continue;

    let searchedAt = null;

    for (let j = i - 1; j >= Math.max(0, i - 8); j--) {
      const timeMatch = lines[j].match(
        /(上午|下午)\s*\d{1,2}:\d{2}/
      );

      if (timeMatch) {
        searchedAt = lines[j];
        break;
      }
    }

    results.push({
      query: match[1].trim(),
      searched_at: `${TARGET_DATE} ${searchedAt || ""}`.trim(),
      source_type: "google_search_history",
      user_visible: true
    });
  }

  const uniqueResults = [
    ...new Map(
      results.map(item => [
        `${item.searched_at}_${item.query}`,
        item
      ])
    ).values()
  ];

  const output = JSON.stringify(uniqueResults, null, 2);

  if (typeof copy === "function") {
    copy(output);
  } else {
    navigator.clipboard?.writeText(output);
  }

  console.log(`Collected ${uniqueResults.length} Google searches.`);
  console.log(output);
})();