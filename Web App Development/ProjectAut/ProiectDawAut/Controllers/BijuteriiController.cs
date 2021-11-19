using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;
using System.Web.Mvc;
using ProiectDawAut.Models;

namespace ProiectDaw.Controllers
{
    [Authorize(Roles = "Admin")]
    public class BijuteriiController : Controller
    {
        private ApplicationDbContext db = new ApplicationDbContext();
        // GET: Bijuterii
        [HttpGet]
        public ActionResult New() //ia bijuteira cu tot cu comenzile ei
        {
            Bijuterii bijuterie = new Bijuterii();
            bijuterie.DistribuitorsList = GetAllDistribuitors();
            bijuterie.Comenzi = new List<Comenzi>();
            return View(bijuterie);
        }
        [Authorize(Roles = "Admin")]
        [HttpPost]//actiune de admin
        public ActionResult New(Bijuterii bijuterieRequest)
        {
            try
            {
                bijuterieRequest.DistribuitorsList = GetAllDistribuitors();
                if (ModelState.IsValid)
                {
                    bijuterieRequest.Distribuitor = db.Distribuitors
                    .FirstOrDefault(p => p.DistribuitorId.Equals(1));
                    db.Bijuterii.Add(bijuterieRequest);
                    db.SaveChanges();
                    return RedirectToAction("Index");
                }
                return View(bijuterieRequest);
            }
            catch (Exception e)
            {
                return View(bijuterieRequest);
            }
        }
        [HttpGet]//get id then edit
        public ActionResult Edit(int? id)
        {
            if (id.HasValue)
            {
                Bijuterii bijuterie = db.Bijuterii.Find(id);
                if (bijuterie == null)
                {
                    return HttpNotFound("Nu exista bijuteria cu id-ul dat " + id.ToString());
                }
                bijuterie.DistribuitorsList = GetAllDistribuitors();

                return View(bijuterie);
            }
            return HttpNotFound("Lipseste parametrul id!");
        }

        [HttpPut]
        public ActionResult Edit(int id, Bijuterii bijuterieRequest)
        {
            try
            {
                bijuterieRequest.DistribuitorsList = GetAllDistribuitors();
                if (ModelState.IsValid)
                {
                    Bijuterii bijuterie = db.Bijuterii
                   .Include("Distribuitor")
                    .SingleOrDefault(b => b.IdBijuterie.Equals(id));
                    if (TryUpdateModel(bijuterie))
                    {
                        bijuterie.Nume = bijuterieRequest.Nume;
                        bijuterie.Tip = bijuterieRequest.Tip;
                        bijuterie.Pret = bijuterieRequest.Pret;
                        bijuterie.Image = bijuterieRequest.Image;
                        bijuterie.Distribuitor = bijuterieRequest.Distribuitor;
                        db.SaveChanges();
                    }
                    return RedirectToAction("Index");
                }
                return View(bijuterieRequest);
            }
            catch (Exception e)
            {
                return View(bijuterieRequest);
            }
        }
        [HttpDelete]
        public ActionResult Delete(int id)
        {
            Bijuterii bijuterie = db.Bijuterii.Find(id);
            if (bijuterie != null)
            {
                db.Bijuterii.Remove(bijuterie);
                db.SaveChanges();
                return RedirectToAction("Index");
            }
            return HttpNotFound("Nu am gasit bijuteria cu id-ul  " + id.ToString());
        }
        [HttpGet]
        [AllowAnonymous]
        public ActionResult Details(int? id)
        {
            if (id.HasValue)
            {
                Bijuterii bijuterie = db.Bijuterii.Find(id);
                if (bijuterie != null)
                {
                    return View(bijuterie);
                }
                return HttpNotFound("Nu am gasit bijuteria cu id-ul  " + id.ToString() + "!");
            }
            return HttpNotFound("Lipseste parametrul id!");
        }
        [AllowAnonymous]
        public ActionResult Index()
        {
            List<Bijuterii> bijuterii = db.Bijuterii.Include("Distribuitor").ToList();
            ViewBag.Bijuterii = bijuterii;
            return View();
        }

        [NonAction] // specificam faptul ca nu este o actiune
        public IEnumerable<SelectListItem> GetAllDistribuitors()
        {
            // generam o lista goala
            var selectList = new List<SelectListItem>();
            foreach (var cover in db.Distribuitors.ToList())
            {
                // adaugam in lista elementele necesare pt dropdown
                selectList.Add(new SelectListItem
                {
                    Value = cover.DistribuitorId.ToString(),
                    Text = cover.Nume
                });
            }
            // returnam lista pentru dropdown
            return selectList;
        }


    }
}