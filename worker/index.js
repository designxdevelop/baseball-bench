export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    if (url.pathname === "/" || url.pathname === "/index.html") {
      const assetUrl = new URL("/site/", url);
      return env.ASSETS.fetch(new Request(assetUrl, request));
    }

    if (url.pathname === "/site" || url.pathname === "/site/" || url.pathname === "/site/index.html") {
      return Response.redirect(new URL("/", url), 308);
    }

    return env.ASSETS.fetch(request);
  },
};
