import base64

svg = '''<svg xmlns="http://www.w3.org/2000/svg" width="800" height="800" viewBox="0 0 800 800">
  <defs>
    <style>
      .c-subtle { fill: none; stroke: rgba(52, 211, 153, 0.07); stroke-width: 1.1; }
      .c-index  { fill: none; stroke: rgba(110, 231, 183, 0.12); stroke-width: 1.6; }
      .c-accent { fill: none; stroke: rgba(167, 243, 208, 0.05); stroke-width: 0.9; stroke-dasharray: 4, 3; }
      .c-txt { fill: rgba(110, 231, 183, 0.22); font-family: monospace; font-size: 8.5px; font-weight: bold; }
    </style>
  </defs>
  <!-- Peak 1 Topography (North-West Ridge) -->
  <path class="c-index" d="M -50,150 Q 80,110 180,160 T 360,130 T 520,210 T 700,160 T 850,220" />
  <path class="c-subtle" d="M -50,180 Q 90,140 190,190 T 370,160 T 530,240 T 710,190 T 850,250" />
  <path class="c-subtle" d="M -50,210 Q 100,170 200,220 T 380,190 T 540,270 T 720,220 T 850,280" />
  <path class="c-subtle" d="M -50,240 Q 110,200 210,250 T 390,220 T 550,300 T 730,250 T 850,310" />
  <path class="c-index" d="M -50,270 Q 120,230 220,280 T 400,250 T 560,330 T 740,280 T 850,340" />
  <text x="210" y="278" class="c-txt">1800m</text>
  <text x="570" y="328" class="c-txt">1800m</text>

  <!-- Ridge 2 Complex Contours (Central Valley & Peak) -->
  <path class="c-subtle" d="M 120,-30 C 180,90 280,140 240,260 S 110,380 200,490 S 390,560 330,680 S 180,820 220,850" />
  <path class="c-index" d="M 160,-30 C 220,90 320,130 280,250 S 150,370 240,480 S 430,550 370,670 S 220,810 260,850" />
  <text x="270" y="248" class="c-txt">1600m</text>
  <path class="c-subtle" d="M 200,-30 C 260,90 360,120 320,240 S 190,360 280,470 S 470,540 410,660 S 260,800 300,850" />

  <!-- Enclosed Elevation Peaks (Topographic Summits) -->
  <path class="c-index" d="M 550,420 C 510,360 620,310 680,350 S 760,460 700,510 S 590,480 550,420 Z" />
  <text x="615" y="345" class="c-txt">2200m ▲</text>
  <path class="c-subtle" d="M 570,420 C 535,375 625,335 670,365 S 735,450 690,490 S 605,465 570,420 Z" />
  <path class="c-subtle" d="M 590,420 C 560,390 630,360 660,380 S 710,440 680,470 S 620,450 590,420 Z" />
  <path class="c-accent" d="M 610,420 C 590,400 635,385 650,395 S 685,430 670,450 S 630,440 610,420 Z" />
  <circle cx="640" cy="420" r="2" fill="rgba(110, 231, 183, 0.25)" />

  <!-- South-Western Slopes -->
  <path class="c-subtle" d="M -40,450 Q 80,420 160,490 T 260,620 T 180,780" />
  <path class="c-index" d="M -40,480 Q 90,450 180,520 T 290,650 T 210,810" />
  <text x="95" y="460" class="c-txt">1400m</text>
  <path class="c-subtle" d="M -40,510 Q 100,480 200,550 T 320,680 T 240,840" />
  <path class="c-subtle" d="M -40,540 Q 110,510 220,580 T 350,710 T 270,870" />

  <!-- South-Eastern Valley Contours -->
  <path class="c-subtle" d="M 420,850 C 460,720 580,680 640,730 S 740,690 850,720" />
  <path class="c-index" d="M 450,850 C 490,735 600,695 660,745 S 760,705 850,735" />
  <text x="570" y="715" class="c-txt">1200m</text>
  <path class="c-subtle" d="M 480,850 C 520,750 620,710 680,760 S 780,720 850,750" />

  <!-- Secondary Summit Loop North East -->
  <path class="c-index" d="M 420,110 C 460,40 570,30 610,80 S 630,170 580,200 S 380,180 420,110 Z" />
  <text x="490" y="60" class="c-txt">2000m ▲</text>
  <path class="c-subtle" d="M 440,115 C 475,60 555,50 590,90 S 605,160 565,185 S 405,170 440,115 Z" />
  <path class="c-accent" d="M 465,120 C 490,80 540,75 565,100 S 580,150 550,170 S 440,160 465,120 Z" />
</svg>'''

b64 = base64.b64encode(svg.encode('utf-8')).decode('utf-8')
data_uri = f"data:image/svg+xml;base64,{b64}"
print("Base64 length:", len(data_uri))
with open("topo_contour.svg", "w", encoding="utf-8") as f:
    f.write(svg)
print("Saved topo_contour.svg successfully!")
