const input = document.getElementById("featureInput");
const imgDiv = document.getElementById("images");
const featureListDiv = document.getElementById("featureList");
const featureHeader = document.getElementById("featureHeader");
const endNote = document.getElementById("endNote");

var imgs = [];
var actImgs = [];
for (var i=0; i<25; i++) {
    let holder = document.createElement("div");
    holder.setAttribute("class", "imgHolder");
    imgDiv.appendChild(holder);
    let img = document.createElement("img");
    img.setAttribute("src", "");
    img.setAttribute("alt", `The ${i+1}th highest-activating game state of this feature`);
    img.setAttribute("style", "display: none;");
    img.setAttribute("id", `img${i}`);
    img.setAttribute("title", String(i+1));
    holder.appendChild(img)
    imgs.push(img)

    let img2 = document.createElement("img");
    img.setAttribute("src", "");
    img.setAttribute("alt", `The effect of the ${i+1}th highest activation of this feature`);
    img.setAttribute("style", "display: none;");
    img.setAttribute("id", `actImg${i}`);
    img.setAttribute("title", String(i+1));
    holder.appendChild(img2)
    actImgs.push(img2)
}

const features = {0: 25, 1001: 25, 1002: 25, 1009: 25, 1013: 25, 1016: 25, 102: 25, 1026: 25, 103: 1, 1030: 25, 1031: 25, 1034: 25, 1035: 25, 1039: 25, 1048: 4, 1052: 25, 1056: 1, 1059: 25, 1067: 25, 107: 25, 1076: 25, 1079: 25, 1080: 25, 1083: 25, 1085: 25, 1088: 25, 1093: 25, 1094: 25, 1097: 25, 1098: 25, 1099: 1, 1100: 25, 1102: 25, 1104: 25, 1105: 25, 1108: 25, 1112: 1, 1114: 25, 1125: 25, 1127: 25, 1132: 25, 1141: 25, 1148: 25, 1151: 25, 1152: 25, 1155: 25, 1156: 8, 1157: 25, 1159: 25, 1161: 1, 1171: 25, 1173: 25, 1179: 25, 1184: 25, 1186: 25, 1188: 3, 1190: 25, 1194: 25, 1202: 25, 1203: 25, 1206: 1, 1210: 25, 1220: 25, 1222: 25, 1223: 25, 1227: 25, 1230: 25, 1231: 25, 1243: 25, 1248: 2, 1256: 25, 1258: 25, 126: 1, 1261: 25, 1262: 25, 1265: 25, 1268: 25, 1272: 25, 1275: 25, 129: 25, 1293: 1, 1294: 25, 1296: 25, 1297: 25, 1300: 25, 1302: 4, 1306: 25, 1307: 6, 1312: 25, 1313: 25, 1317: 25, 1321: 25, 1331: 25, 1346: 25, 1349: 5, 135: 25, 1356: 25, 1359: 1, 1371: 25, 1387: 25, 139: 1, 1392: 4, 1394: 25, 1398: 25, 1404: 4, 1405: 25, 1409: 12, 1411: 25, 1413: 25, 1420: 24, 1422: 1, 1427: 25, 1429: 25, 1434: 5, 1439: 25, 1443: 25, 1448: 25, 1453: 25, 1457: 1, 146: 25, 1468: 25, 1470: 25, 1475: 25, 1480: 25, 1496: 25, 1497: 25, 1503: 25, 1507: 1, 1510: 25, 1512: 25, 1513: 25, 1514: 25, 1519: 25, 152: 25, 1530: 25, 1533: 1, 1535: 3, 1540: 25, 1545: 1, 1553: 25, 1558: 25, 1562: 7, 1573: 25, 158: 1, 1581: 25, 1585: 25, 1596: 25, 1597: 3, 1598: 1, 1602: 25, 1608: 25, 1611: 25, 1615: 25, 1618: 25, 1622: 2, 1623: 8, 1628: 25, 1631: 25, 1633: 25, 1636: 25, 1637: 25, 1643: 25, 1644: 25, 1651: 25, 1662: 25, 1663: 25, 1664: 25, 1665: 25, 1666: 25, 1667: 25, 1684: 25, 1687: 11, 169: 2, 1694: 6, 1695: 1, 1698: 1, 170: 1, 1700: 25, 1713: 25, 1719: 1, 1727: 25, 1728: 25, 1736: 25, 1738: 25, 1747: 1, 1748: 25, 1749: 25, 1751: 25, 1753: 25, 1757: 25, 1761: 25, 1763: 25, 1769: 25, 1770: 25, 1780: 3, 1783: 25, 1784: 1, 1796: 25, 1797: 25, 1801: 25, 1806: 25, 1807: 25, 1809: 1, 1812: 4, 1813: 3, 1815: 25, 1818: 25, 1820: 25, 1824: 2, 1830: 25, 1832: 25, 1835: 25, 1836: 25, 1838: 25, 1840: 25, 1850: 25, 1852: 25, 1858: 25, 1865: 25, 1873: 25, 1876: 1, 1881: 25, 1884: 25, 1887: 1, 1889: 25, 1895: 25, 1896: 25, 190: 25, 1902: 25, 1903: 3, 1909: 25, 1917: 1, 1920: 2, 1922: 25, 1926: 4, 193: 1, 1930: 25, 1938: 1, 1944: 3, 1949: 2, 1955: 1, 196: 25, 1960: 25, 1962: 25, 1969: 25, 197: 25, 1973: 25, 1976: 25, 1985: 25, 1998: 15, 1999: 1, 2002: 25, 2005: 25, 2016: 25, 2023: 2, 2027: 25, 2030: 25, 2036: 25, 2040: 6, 205: 1, 21: 1, 213: 25, 215: 1, 22: 25, 221: 25, 223: 19, 224: 25, 226: 25, 24: 25, 240: 11, 244: 25, 246: 25, 251: 25, 26: 25, 270: 25, 276: 25, 277: 3, 278: 25, 301: 25, 302: 25, 309: 25, 313: 25, 315: 25, 316: 17, 318: 25, 319: 25, 322: 25, 327: 25, 335: 25, 336: 25, 348: 14, 350: 25, 351: 25, 352: 1, 356: 25, 36: 25, 366: 25, 367: 4, 368: 1, 372: 25, 375: 2, 377: 25, 378: 17, 379: 25, 381: 2, 382: 25, 383: 25, 385: 25, 39: 25, 394: 25, 398: 25, 400: 1, 404: 25, 406: 25, 407: 25, 408: 25, 411: 25, 412: 25, 413: 24, 414: 25, 416: 25, 42: 25, 425: 25, 428: 25, 43: 1, 430: 3, 439: 1, 44: 25, 450: 1, 454: 25, 46: 4, 462: 25, 463: 3, 466: 25, 467: 25, 47: 1, 476: 25, 477: 25, 485: 25, 486: 2, 494: 25, 496: 2, 504: 25, 506: 25, 517: 25, 519: 2, 523: 25, 527: 25, 532: 25, 536: 25, 541: 25, 543: 25, 548: 25, 551: 25, 553: 25, 56: 25, 561: 6, 563: 1, 571: 5, 584: 1, 592: 25, 595: 25, 597: 17, 6: 1, 611: 3, 619: 25, 623: 3, 624: 25, 625: 13, 63: 25, 630: 25, 631: 25, 633: 1, 638: 25, 641: 1, 643: 25, 647: 4, 661: 25, 663: 25, 667: 25, 676: 25, 679: 25, 680: 25, 682: 25, 684: 1, 689: 25, 695: 1, 699: 25, 7: 25, 701: 1, 702: 25, 709: 1, 71: 1, 710: 25, 716: 25, 719: 1, 721: 25, 726: 25, 727: 4, 73: 25, 731: 25, 734: 3, 737: 3, 748: 25, 75: 25, 753: 25, 754: 25, 759: 25, 761: 25, 767: 2, 768: 25, 774: 14, 775: 25, 777: 1, 784: 25, 792: 25, 796: 25, 799: 25, 803: 1, 806: 25, 81: 25, 811: 1, 820: 25, 823: 1, 824: 25, 825: 25, 833: 25, 835: 25, 836: 1, 837: 4, 839: 25, 84: 3, 854: 1, 855: 25, 857: 4, 858: 1, 863: 25, 874: 25, 875: 25, 876: 1, 883: 25, 884: 25, 892: 25, 896: 1, 899: 25, 906: 25, 907: 25, 914: 25, 917: 25, 92: 1, 927: 25, 928: 25, 932: 1, 941: 1, 943: 25, 949: 25, 95: 1, 954: 2, 956: 2, 962: 25, 967: 1, 968: 25, 976: 25, 980: 25, 998: 25, 999: 16}; // Baking this data is kind of a hack, but it's so short (only a few hundred elements) that it's hopefully not a problem.

const feats = Object.keys(features).sort((a, b) => parseInt(a)-parseInt(b))
for (var num in feats) {
    let feat = feats[num]
    let btn = document.createElement("button");
    btn.textContent = feat;
    btn.addEventListener("click", () => showFeature(feat));
    featureListDiv.appendChild(btn);
}

function showFeature(featureId) {
    if (featureId in features) {
        featureHeader.textContent = `Feature Viewer: Feature ${featureId}`;

        const numImages = features[featureId]
        for (var i = 0; i < numImages; i++) {
            const imagePath = `imgs/feat_${featureId}/${i}.png`;

            let img = imgs[i];

            img.src = imagePath;
            img.style.display = "inline-block";
            img.style.visibility = "visible";

            img.onerror = function () {
            img.style.display = "none";
            alert(`Image "${imagePath}" not found.`);
            };

            const imagePath2 = `imgs/feat_${featureId}/act_${i}.png`;

            let img2 = actImgs[i];

            img2.src = imagePath2;
            img2.style.display = "inline-block";
            img2.style.visibility = "visible";

            img2.onerror = function () {
            img2.style.display = "none";
            alert(`Image "${imagePath}" not found.`);
            };
        }

        if (numImages < 25) {
            for (var i = numImages; i < 25; i++) {
                imgs[i].style.visibility = "hidden";
                actImgs[i].style.visibility = "hidden";
            }
            endNote.innerHTML = `Only ${numImages} game states were displayed, because this feature only activates on ${numImages} game states (over 10,000 games)!`;
        } else {
            endNote.innerHTML = ""; // TODO maybe add text about "this feature activates X% of the time"?
        }

    } else {
        alert(`${featureId} is not a valid feature number!`);
    }
}

input.addEventListener("keydown", function(event) {
  if (event.key === "Enter") {
    const featureId = input.value.trim();
    showFeature(featureId);
  }
});