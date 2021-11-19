using Microsoft.SqlServer.Server;
using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;
using System.Data.Entity;
using System.Linq;
using System.Web;

namespace ProiectDaw.Models
{
    public class Bijuterii
    {
        [Key, Column("Id_Bijuterie")]
        public int IdBijuterie { get; set; }
        public string Nume { get; set; }
        public string Tip { get; set; }

        public int Pret { get; set; }
        public string Image { get; set; }

        // many to many
        public virtual ICollection<Comenzi> Comenzi { get; set; }
    }
    public class DbCtx : DbContext
    {
        public DbCtx() : base("DbConnectionString")
        {
            Database.SetInitializer<DbCtx>(new Initp());
            //Database.SetInitializer<DbCtx>(new CreateDatabaseIfNotExists<DbCtx>());
            //Database.SetInitializer<DbCtx>(new DropCreateDatabaseIfModelChanges<DbCtx>());
            //Database.SetInitializer<DbCtx>(new DropCreateDatabaseAlways<DbCtx>());
        }
        public DbSet<Bijuterii> Bijuterii { get; set; }
        public DbSet<User> Users { get; set; }
        public DbSet<Comenzi> Comenzi { get; set; }
        public DbSet<ContactInfo> ContactInfo { get; set; }


    }

    public class Initp : DropCreateDatabaseAlways<DbCtx>
    {
        protected override void Seed(DbCtx ctx)
        {
            ctx.Bijuterii.Add(new Bijuterii
            {
                Tip = "Inel",
                Nume = "PRINCESS DREAM",
                Image = "https://charm.ro/pub/media/catalog/product/cache/74c1057f7991b4edb2bc7bdaa94de933/s/c/scr066_2__2nd.jpg",
                Pret = 69
            });
            ctx.Bijuterii.Add(new Bijuterii
            {
                Tip = "Inel",
                Nume = " FLOWER DANCE",
                Image = "https://charm.ro/pub/media/catalog/product/cache/74c1057f7991b4edb2bc7bdaa94de933/s/c/scr390_2nd.jpg",
                Pret = 119
            });
            ctx.Bijuterii.Add(new Bijuterii
            {
                Tip = "Cercei",
                Nume = "NOBLE LIGHT",
                Image = "https://charm.ro/pub/media/catalog/product/cache/74c1057f7991b4edb2bc7bdaa94de933/s/c/sce358_2nd.jpg",
                Pret = 109
            });
            ctx.Bijuterii.Add(new Bijuterii
            {
                Tip = "Lantisor",
                Nume = "MOON AND SUN",
                Image = "https://charm.ro/pub/media/catalog/product/cache/74c1057f7991b4edb2bc7bdaa94de933/s/c/scn272_2nd.jpg",
                Pret = 139
            });

            ctx.Users.Add(new User
            {

            });
            ctx.SaveChanges();
            base.Seed(ctx);
        }
    }
}
