using System.Data.Entity;
using System.Security.Claims;
using System.Threading.Tasks;
using Microsoft.AspNet.Identity;
using Microsoft.AspNet.Identity.EntityFramework;


namespace ProiectDawAut.Models
{
    // You can add profile data for the user by adding more properties to your ApplicationUser class, please visit https://go.microsoft.com/fwlink/?LinkID=317594 to learn more.
    public class ApplicationUser : IdentityUser
    {
        public async Task<ClaimsIdentity> GenerateUserIdentityAsync(UserManager<ApplicationUser> manager)
        {
            // Note the authenticationType must match the one defined in CookieAuthenticationOptions.AuthenticationType
            var userIdentity = await manager.CreateIdentityAsync(this, DefaultAuthenticationTypes.ApplicationCookie);
            // Add custom user claims here
            return userIdentity;
        }
    }

    public class ApplicationDbContext : IdentityDbContext<ApplicationUser>
    {
        public ApplicationDbContext()
            : base("DefaultConnection", throwIfV1Schema: false)
        {
            Database.SetInitializer<ApplicationDbContext>(new Initp());
        }
        public DbSet<Bijuterii> Bijuterii { get; set; }
    
        public DbSet<Comenzi> Comenzi { get; set; }
        public DbSet<ContactInfo> ContactInfo { get; set; }
        public DbSet<Distribuitor> Distribuitors { get; set; }
        public static ApplicationDbContext Create()
        {
            return new ApplicationDbContext();
        }

    }
    public class Initp : DropCreateDatabaseAlways<ApplicationDbContext>
    {
        protected override void Seed(ApplicationDbContext ctx)
        {
            ctx.Bijuterii.Add(new Bijuterii
            {
                Tip = "Inel",
                Nume = "PRINCESS DREAM",
                Image = "https://charm.ro/pub/media/catalog/product/cache/74c1057f7991b4edb2bc7bdaa94de933/s/c/scr066_2__2nd.jpg",
                Pret = 70,
                Distribuitor = new Distribuitor { Nume = "Charms", ContactInfo = new ContactInfo { PhoneNumber = "0764343228",  Adresa = "Bucuresti, str. Narciselor nr. 30", StartYear = 2013, StartMonth= "02", StartDay ="21"} }
            });
            ctx.Bijuterii.Add(new Bijuterii
            {
                Tip = "Inel",
                Nume = " FLOWER DANCE",
                Image = "https://charm.ro/pub/media/catalog/product/cache/74c1057f7991b4edb2bc7bdaa94de933/s/c/scr390_2nd.jpg",
                Pret = 120,
                Distribuitor = new Distribuitor { Nume = "Pandora", ContactInfo = new ContactInfo { PhoneNumber = "0712345878" , Adresa = "Bucuresti, str. Lalelelor nr. 30", StartYear = 2011, StartMonth = "03", StartDay = "27" } }
            });
            ctx.Bijuterii.Add(new Bijuterii
            {
                Tip = "Cercei",
                Nume = "NOBLE LIGHT",
                Image = "https://charm.ro/pub/media/catalog/product/cache/74c1057f7991b4edb2bc7bdaa94de933/s/c/sce358_2nd.jpg",
                Pret = 110,
                Distribuitor = new Distribuitor { Nume = "Teilor", ContactInfo = new ContactInfo { PhoneNumber = "0711345678", Adresa = "Cluj-Napoca, str. Eroilor nr. 21", StartYear = 2015, StartMonth = "06", StartDay = "14" } }
            });
            ctx.Bijuterii.Add(new Bijuterii
            {
                Tip = "Lantisor",
                Nume = "MOON AND SUN",
                Image = "https://charm.ro/pub/media/catalog/product/cache/74c1057f7991b4edb2bc7bdaa94de933/s/c/scn272_2nd.jpg",
                Pret = 140,
                Distribuitor = new Distribuitor { Nume = "Calliope", ContactInfo = new ContactInfo { PhoneNumber = "0712665678", Adresa = "Bucuresti, str. Revolutiei nr. 12", StartYear = 2018, StartMonth = "08", StartDay = "21" } }
            });
            
            ctx.SaveChanges();
            base.Seed(ctx);
        }
    }
}