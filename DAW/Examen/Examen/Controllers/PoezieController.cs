using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;
using System.Web.Mvc;
using Examen.Models;
using Examen.Models.MyDatabaseInitializer;

namespace Examen.Controllers
{
    public class PoezieController : Controller
    {
        // GET: Poezie
        private DbCtx db = new DbCtx();
        public ActionResult Index()
        {
            List<Poezie> poezii = db.Poezii.Include("Volum").ToList();
            ViewBag.Poezii = poezii;
            return View();
        }
        [HttpGet]
        public ActionResult New()
        {
            Poezie poezie = new Poezie();
            poezie.VolumeList = GetAllVolumes();          
            return View(poezie);
        }

        [HttpPost]
        public ActionResult New(Poezie poezieRequest)
        {
            poezieRequest.VolumeList = GetAllVolumes();
            try
            {
                if (ModelState.IsValid)
                {
                    poezieRequest.Volum = db.Volume
                    .FirstOrDefault(b => b.VolumId.Equals(1));
                   // poezieRequest.VolumId = poezieRequest.Volum.VolumId;
                    db.Poezii.Add(poezieRequest);
                    db.SaveChanges();
                    return RedirectToAction("Index");
                }
                return View(poezieRequest);
            }
            catch (Exception e)
            {
                var msg = e.Message;
                return View(poezieRequest);
            }
        }
        [HttpGet]
        public ActionResult Edit(int? id)
        {
            if (id.HasValue)
            {
                Poezie poezie = db.Poezii.Find(id);
                if (poezie == null)
                {
                    return HttpNotFound("Couldn't find the poetry with id " + id.ToString());
                }
                poezie.VolumeList = GetAllVolumes();
                return View(poezie);
            }
            return HttpNotFound("Missing poem id parameter!");
        }
        [HttpPut]
        public ActionResult Edit(int id, Poezie poezieRequest)
        {
            try
            {
                poezieRequest.VolumeList = GetAllVolumes();
                if (ModelState.IsValid)
                {
                    Poezie poezie = db.Poezii
                    .Include("Volum")
                   .SingleOrDefault(b => b.PoezieId.Equals(id));
                    if (TryUpdateModel(poezie))
                    {
                        poezie.Titlu = poezieRequest.Titlu;
                        poezie.Autor = poezieRequest.Autor;
                        poezie.NumarStrofe = poezieRequest.NumarStrofe;
                        //poezie.VolumId= poezieRequest.VolumId;
                        db.SaveChanges();
                    }
                    return RedirectToAction("Index");
                }
                return View(poezieRequest);
            }
            catch (Exception e)
            {
                return View(poezieRequest);
            }
        }
        [HttpDelete]
        public ActionResult Delete(int id)
        {
            Poezie poezie = db.Poezii.Find(id);
            if (poezie != null)
            {
                db.Poezii.Remove(poezie);
                db.SaveChanges();
                return RedirectToAction("Index");
            }
            return HttpNotFound("Couldn't find the poetry with id " + id.ToString() + "!");
        }
        [NonAction]
        public IEnumerable<SelectListItem> GetAllVolumes()
        {
            var selectList = new List<SelectListItem>();
            foreach (var volum in db.Volume.ToList())
            {
                selectList.Add(new SelectListItem
                {
                    Value = volum.VolumId.ToString(),
                    Text = volum.Denumire
                });
            }
            return selectList;
        }

        public ActionResult CautareSubstring(string cuvant)
        {
            List<Poezie> poezii = db.Poezii.Include("Volum").ToList();
            List<Poezie> poeziiSubstr = new List<Poezie>();
            if (cuvant != null)
            {
                foreach (Poezie poezie in poezii) {
                    if ((poezie.Titlu).Contains(cuvant))
                        poeziiSubstr.Add(poezie);
                }
                ViewBag.Poezii = poeziiSubstr;
                return View();
            }
            return HttpNotFound("Cuvant null!");
        }
        public ActionResult CautareSubstringVolume(string cuvant)
        {
            List<Poezie> poezii = db.Poezii.Include("Volum").ToList();
            List<Poezie> poeziiSubstr = new List<Poezie>();
            if (cuvant != null)
            {
                foreach (Poezie poezie in poezii)
                {
                    if ((poezie.Volum.Denumire).Contains(cuvant))
                        poeziiSubstr.Add(poezie);
                }
                ViewBag.Poezii = poeziiSubstr;
                return View();
            }
            return HttpNotFound("Cuvant null!");
        }

    }
}