(async () => {
  // auto_fetch.py sets window.__PODCAST_FETCH__ to run unattended.
  const AUTO = window.__PODCAST_FETCH__;
  const TARGET_COUNT = AUTO?.count || 100;
  const WAIT_MS = 1000;
  const items = new Map();

  const sleep = ms =>
    new Promise(resolve => setTimeout(resolve, ms));
  
  const MUSIC_CONTENT_PATTERNS = [
    /\b(?:official\s+)?(?:music\s+video|mv|m\/v)\b/i,
    /\b(?:official\s+audio|audio\s+only|lyrics?(?:\s+video)?|visuali[sz]er)\b/i,
    /\b(?:dance\s+practice|dance\s+performance|live\s+performance|stage\s+mix|fancam|focus\s+cam|karaoke|instrumental|remix|song\s+cover|cover\s+by)\b/i,
    /\b(?:sped[\s-]?up|slowed(?:\s+down)?)\b/i,
    /뮤직비디오|가사\s*(?:영상|버전)|안무\s*영상|공식\s*음원/,
    /ミュージックビデオ|歌詞付き|ダンスプラクティス|公式音源|カラオケ/,
    /音樂錄影帶|歌詞(?:版|影片)|舞蹈練習|官方音源|卡拉OK/
  ];

  function isMusicContent(title) {
    return MUSIC_CONTENT_PATTERNS.some(pattern => pattern.test(title));
  }

  function collectVideos() {
    const anchors = document.querySelectorAll(
      'a[href^="/watch"], a[href^="/shorts/"]'
    );

    for (const a of anchors) {
      const url = new URL(a.href);
      if (a.closest("ytd-ad-slot-renderer, ytd-in-feed-ad-layout-renderer, ytd-promoted-video-renderer")) continue;
      const card = a.closest(
        [
          "ytd-rich-grid-media",
          "ytd-rich-item-renderer",
          "ytd-video-renderer",
          "ytd-reel-item-renderer",
          "ytd-lockup-view-model",
          "ytd-compact-video-renderer"
        ].join(",")
      );

      const isWatch =
        url.pathname === "/watch" &&
        url.searchParams.get("v");

      const isShort =
        url.pathname.startsWith("/shorts/") &&
        url.pathname.split("/")[2];

      if (!isWatch && !isShort) continue;

      const cardText = (card?.innerText || "").trim();

      const isPlaylist =
        url.searchParams.has("list") ||
        /Mix|Playlist|合輯|播放清單/i.test(cardText);

      if (isPlaylist) continue;
      

      const titleElement = card?.querySelector(
        [
          "#video-title",
          "#video-title-link",
          "a#video-title",
          "h3 a",
          "yt-lockup-metadata-view-model a"
        ].join(",")
      );

      const titleCandidates = [
        titleElement?.textContent,
        titleElement?.getAttribute("title"),
        a.getAttribute("title")
      ];

      let title = titleCandidates
        .map(value => (value || "").trim())
        .find(Boolean) || "";

      if (!title) {
        const lines = cardText
          .split("\n")
          .map(line => line.trim())
          .filter(Boolean);

        title = lines.find(line =>
          !/^\d{1,2}:\d{2}(:\d{2})?$/.test(line) &&
          !/^(觀看|Watch|Shorts)$/i.test(line)
        ) || "";
      }

      const channel = (
        card?.querySelector(
          [
            "#channel-name a",
            "#channel-name",
            "ytd-channel-name a",
            "yt-formatted-string#text"
          ].join(",")
        )?.textContent || ""
      ).trim();

      if (!title) continue;
      if (/^(Sponsored|贊助|廣告|Ad)$/im.test(cardText)) continue;
      if (/^\d{1,2}:\d{2}(:\d{2})?$/.test(title)) continue;
      if (/^(觀看|Watch|Shorts)$/i.test(title)) continue;
      if (isMusicContent(title)) continue;
      if (!items.has(url.href)) {
        items.set(url.href, {
          title,
          channel,
          url: url.href
        });
      }
    }
  }

  let previousHeight = 0;
  let unchangedCount = 0;

  for (let i = 0; i < 150; i++) {
    collectVideos();

    console.log(`Collected ${items.size} videos.`);

    if (items.size >= TARGET_COUNT) break;

    window.scrollBy(0, window.innerHeight);
    await sleep(WAIT_MS);

    const currentHeight = document.documentElement.scrollHeight;

    if (currentHeight === previousHeight) {
      unchangedCount++;
    } else {
      unchangedCount = 0;
    }

    if (unchangedCount >= 5) break;

    previousHeight = currentHeight;
  }

  collectVideos();

  const result = [...items.values()].slice(0, TARGET_COUNT);
  const output = JSON.stringify(result, null, 2);

  if (AUTO) return result;

  if (typeof copy === "function") {
    copy(output);
  } else {
    await navigator.clipboard?.writeText(output);
  }

  console.log(`Copied ${result.length} YouTube videos.`);
  console.log(output);
})();