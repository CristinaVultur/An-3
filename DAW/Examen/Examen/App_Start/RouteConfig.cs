using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;
using System.Web.Mvc;
using System.Web.Routing;

namespace Examen
{
    public class RouteConfig
    {
        public static void RegisterRoutes(RouteCollection routes)
        {
            routes.IgnoreRoute("{resource}.axd/{*pathInfo}");

            routes.MapRoute(name: "CautareSubstringVolume", url: "volume/{cuvant}",
                defaults: new
                {
                    controller = "Poezie",
                    action = "CautareSubstringVolume",
                    cuvant = UrlParameter.Optional
                });
            routes.MapRoute( name: "CautareSubstring", url: "poezii/{cuvant}",
                defaults: new
                {
                    controller = "Poezie",
                    action = "CautareSubstring",
                    cuvant = UrlParameter.Optional
                } );
            

            routes.MapRoute(
                name: "Default",
                url: "{controller}/{action}/{id}",
                defaults: new { controller = "Home", action = "Index", id = UrlParameter.Optional }
            );
        }
    }
}
