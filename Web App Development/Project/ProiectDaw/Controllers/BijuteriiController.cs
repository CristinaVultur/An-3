using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;
using System.Web.Mvc;
using ProiectDaw.Models;

namespace ProiectDaw.Controllers
{
    public class BijuteriiController : Controller
    {
        // GET: Bijuterii
        [HttpGet]
        public ActionResult New() //ia bijuteira cu tot cu comenzile ei
        {
            Bijuterii bijuterie = new Bijuterii();
            bijuterie.Comenzi = new List<Comenzi>();
            return View(bijuterie);
        }
        [HttpPost]//actiune de admin
        public ActionResult New(Bijuterii bijuterieRequest)
        {
            try
            {
                if (ModelState.IsValid)
                {
                    //bijuterieRequest.Publisher = db.Publishers
                   //.FirstOrDefault(p => p.PublisherId.Equals(1));
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
                return View(bijuterie);
            }
            return HttpNotFound("Lipseste parametrul id!");
        }

        [HttpPut]
        public ActionResult Edit(int id, Bijuterii bijuterieRequest)
        {
            try
            {
                if (ModelState.IsValid)
                {
                    Bijuterii bijuterie = db.Bijuterii
                   .Include("Publisher")
                    .SingleOrDefault(b => b.IdBijuterie.Equals(id));
                    if (TryUpdateModel(bijuterie))
                    {
                        bijuterie.Nume = bijuterieRequest.Nume;
                        bijuterie.Tip = bijuterieRequest.Tip;
                        bijuterie.Pret = bijuterieRequest.Pret;
                        bijuterie.Image = bijuterieRequest.Image;
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
        private DbCtx db = new DbCtx();
        public ActionResult Index()
        {
            List<Bijuterii> bijuterii = db.Bijuterii.ToList();
            ViewBag.Bijuterii = bijuterii;
            return View();
        }
    }
    

}