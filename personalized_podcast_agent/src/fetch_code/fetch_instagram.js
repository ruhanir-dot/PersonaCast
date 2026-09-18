(async () => {
    const TARGET_COUNT = 100;
    const WAIT_MS = 1000;
    const items = new Map();

    const sleep = (ms) =>
        new Promise(resolve => setTimeout(resolve, ms));

    function collect() {
        for (const a of document.querySelectorAll(
            'a[href^="/p/"], a[href^="/reel/"]'
        )) {
            const url = new URL(a.href);
            url.search = "";

            const article = a.closest("article");
            const text = (article?.innerText || "").trim();

            if (/Sponsored|贊助|廣告|Paid partnership/i.test(text)) {
                continue;
            }

            if (!items.has(url.href)) {
                const lines = text
                    .split("\n")
                    .map(x => x.trim())
                    .filter(Boolean);

                const username =
                    article?.querySelector('header a[href^="/"]')
                        ?.getAttribute("href")
                        ?.split("/")
                        .filter(Boolean)[0] || "";

                const hashtags = [
                    ...new Set(text.match(/#[\p{L}\p{N}_]+/gu) || [])
                ];

                items.set(url.href, {
                    title: lines[0] || "",
                    text,
                    url: url.href,
                    source_type: url.pathname.startsWith("/reel/")
                        ? "instagram_reel"
                        : "instagram_post"
                });
            }
        }
    }

    for (let i = 0; i < 100 && items.size < TARGET_COUNT; i++) {
        collect();

        if (items.size >= TARGET_COUNT) break;

        window.scrollBy(0, window.innerHeight);
        await sleep(WAIT_MS);
    }

    collect();

    const result = [...items.values()].slice(0, TARGET_COUNT);
    const output = JSON.stringify(result, null, 2);

    const blob = new Blob([output], {
        type: "application/json"
    });

    const downloadLink = document.createElement("a");
    downloadLink.href = URL.createObjectURL(blob);
    downloadLink.download = "instagram_feed_manual.json";

    document.body.appendChild(downloadLink);
    downloadLink.click();
    downloadLink.remove();

    URL.revokeObjectURL(downloadLink.href);

    console.log(`Downloaded ${result.length} Instagram feeds.`);
    console.log(output);
})();