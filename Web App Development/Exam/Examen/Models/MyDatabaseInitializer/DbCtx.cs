using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;
using System.Data.Entity;

namespace Examen.Models.MyDatabaseInitializer
{
    public class DbCtx : DbContext
    {
        public DbCtx() : base("DbConnectionString")
        {
            Database.SetInitializer<DbCtx>(new Initp());
            
        }
        public DbSet<Poezie> Poezii { get; set; }
        public DbSet<Volum> Volume { get; set; }

        public class Initp : DropCreateDatabaseAlways<DbCtx>
        { 
            protected override void Seed(DbCtx ctx)
            {
                Volum volum1 = new Volum { VolumId = 1, Denumire = "Volum 1" };
                Volum volum2 = new Volum { VolumId = 2, Denumire = "Volum 2" };
                Volum volum3 = new Volum { VolumId = 3, Denumire = "Volum 3" };

                ctx.Volume.Add(volum1);
                ctx.Volume.Add(volum2);
                ctx.Volume.Add(volum3);

                ctx.Poezii.Add(new Poezie
                {
                    Titlu = "Poezie 1",
                    Autor = "Autor 1",
                    NumarStrofe = 10,
                    Volum = volum1
                });
                ctx.Poezii.Add(new Poezie
                {
                    Titlu = "Poezie 2",
                    Autor = "Autor 2",
                    NumarStrofe = 15,
                    Volum = volum1
                });
                ctx.Poezii.Add(new Poezie
                {
                    Titlu = "Poezie 3",
                    Autor = "Autor 3",
                    NumarStrofe = 10,
                    Volum = volum2
                });
                ctx.Poezii.Add(new Poezie
                {
                    Titlu = "Poezie 4",
                    Autor = "Autor 4",
                    NumarStrofe = 13,
                    Volum = volum2
                });
                ctx.Poezii.Add(new Poezie
                {
                    Titlu = "Poezie 5",
                    Autor = "Autor 5",
                    NumarStrofe = 9,
                    Volum = volum3
                });
                ctx.Poezii.Add(new Poezie
                {
                    Titlu = "Poezie 51",
                    Autor = "Autor 5",
                    NumarStrofe = 4,
                    Volum = volum3
                });

                ctx.SaveChanges();
                base.Seed(ctx);
            }
        }

    }
}