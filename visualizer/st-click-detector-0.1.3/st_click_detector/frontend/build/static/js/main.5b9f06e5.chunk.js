(this.webpackJsonpstreamlit_component_template =
  this.webpackJsonpstreamlit_component_template || []).push([
  [0],
  {
    5: function (e, t, n) {
      e.exports = n(6);
    },
    6: function (e, t, n) {
      "use strict";
      n.r(t);
      var o = n(2),
        a = n(4),
        m = document.createElement("link"),
        s1 = document.createElement("script"),
        s2 = document.createElement("script");
      (m.rel = "stylesheet"),
        (m.href ="//cdnjs.cloudflare.com/ajax/libs/highlight.js/11.7.0/styles/default.min.css"),
        //   "//cdnjs.cloudflare.com/ajax/libs/highlight.js/11.7.0/styles/base16/default-dark.min.css"),
        document.head.appendChild(m);
      s1.src =
        "//cdnjs.cloudflare.com/ajax/libs/highlight.js/11.7.0/highlight.min.js";
      document.head.appendChild(s1);
      s2.innerHTML = "hljs.highlightAll();";
      document.head.appendChild(s2);
      var A =
        "<img src='data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII='>";
      o.a.events.addEventListener(o.a.RENDER_EVENT, function (e) {
        var t = e.detail,
          n = document.body.lastElementChild;
        n && document.body.removeChild(n);
        var m = document.body.appendChild(document.createElement("div"));
        new a.a(function (e, t) {
          o.a.setFrameHeight();
        }).observe(document.body),
          (m.innerHTML = A + t.args.html_content + A);
        for (
          var s = document.getElementsByTagName("a"),
            c = function (e) {
              "" !== s[e].id &&
                (s[e].onclick = function () {
                  o.a.setComponentValue(s[e].id);
                });
                // (s[e].onmouseover = function () {
                //   for (const child of s[e].children) {
                //     var mystyle = child.getAttribute("style")
                //     console.log(mystyle);
                //     if (!mystyle.includes("auto")) {
                //         spl_index = mystyle.indexOf(";overflow-x")
                //         mystyle = mystyle.substring(0, spl_index) 
                //             + ";overflow-x:auto" + mystyle.substring(spl_index+11)
                //         child.setAttribute("style", mystyle);
                //         var m = document.body.appendChild(document.createElement("div"));
                //     }
                //   }
                // });
                // (s[e].onmouseleave = function () {
                //   for (const child of s[e].children) {
                //     var mystyle = child.getAttribute("style")
                //     console.log(mystyle);
                //     if (mystyle.includes("auto")) {
                //         spl_index = mystyle.indexOf(";overflow-x")
                //         mystyle = mystyle.substring(0, spl_index) 
                //             + ";overflow-x:hidden" + mystyle.substring(spl_index+11)
                //         child.setAttribute("style", mystyle);
                //     }
                //   }
                // });
            },
            l = 0;
          l < s.length;
          l++
        )
          c(l);
        hljs.highlightAll();
      }),
        o.a.setComponentReady();
    },
  },
  [[5, 1, 2]],
]);
//# sourceMappingURL=main.5b9f06e5.chunk.js.map
