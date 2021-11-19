using ProiectDawAut.Models;

using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;
using System.Web.Mvc;

namespace ProiectDawAut.Controllers
{
    [Authorize(Roles = "Admin")]
    public class DistribuitorController : Controller
    {
        private ApplicationDbContext ctx = new ApplicationDbContext();

        // GET: Publisher
        public ActionResult Index()
        {
            ViewBag.Distribuitors = ctx.Distribuitors.ToList();
            return View();
        }

        public ActionResult Details(int? id)
        {
            if (id.HasValue)
            {
                Distribuitor distribuitor = ctx.Distribuitors.Find(id);
                if (distribuitor != null)
                {
                    //ViewBag.Region = ctx.Regions.Find(publisher.ContactInfo.RegionId).Name;
                    return View(distribuitor);
                }
                return HttpNotFound("Couldn't find the publisher with id " + id.ToString() + "!");
            }
            return HttpNotFound("Missing publisher id parameter!");
        }

        [HttpGet]
        public ActionResult New()
        {        

            DistribuitorContactViewModel pc = new DistribuitorContactViewModel();
            return View(pc);
        }

        [HttpPost]
        public ActionResult New(DistribuitorContactViewModel pcViewModel)
        {
          
            try
            {
                if (ModelState.IsValid)
                {
                    ContactInfo contact = new ContactInfo
                    {
                        PhoneNumber = pcViewModel.PhoneNumber,                      
                        StartDay = pcViewModel.StartDay,
                        StartMonth = pcViewModel.StartMonth,
                        StartYear = pcViewModel.StartYear,
                        
                    };
                    // vom adauga in baza de date ambele obiecte 
                    ctx.ContactInfo.Add(contact);
                    Distribuitor publisher = new Distribuitor
                    {
                        Nume = pcViewModel.Nume,
                        ContactInfo = contact
                    };
                    ctx.Distribuitors.Add(publisher);
                    ctx.SaveChanges();
                    return RedirectToAction("Index");
                }
                return View(pcViewModel);
            }
            catch (Exception e)
            {
                var msg = e.Message;
                return View(pcViewModel);
            }
        }

        [HttpDelete]
        public ActionResult Delete(int id)
        {
            Distribuitor distribuitor = ctx.Distribuitors.Find(id);
            ContactInfo contact = ctx.ContactInfo.Find(distribuitor.ContactInfo.ContactInfoId);

            if (distribuitor != null)
            {
                ctx.Distribuitors.Remove(distribuitor);
                ctx.ContactInfo.Remove(contact);
                ctx.SaveChanges();
                return RedirectToAction("Index");
            }
            return HttpNotFound("Couldn't find the publisher with id " + id.ToString() + "!");
        }
        [HttpGet]//get id then edit
        public ActionResult Edit(int? id)
        {
            if (id.HasValue)
            {
                
                
                Distribuitor distribuitor = ctx.Distribuitors.Find(id);
                ContactInfo contact = ctx.ContactInfo.Find(distribuitor.ContactInfo.ContactInfoId);
                if (distribuitor == null)
                {
                    return HttpNotFound("Nu exista bijuteria cu id-ul dat " + id.ToString());
                }
                //distribuitor.Nume = distribuitor.Nume;
               // distribuitor.PhoneNumber = contact.PhoneNumber;
                //distribuitor.StartYear = contact.StartYear;
                //pc.StartMonth = contact.StartMonth;
                //pc.StartDay = contact.StartDay;
                //pc.Adresa = contact.Adresa;

                return View(distribuitor);
            }
            return HttpNotFound("Lipseste parametrul id!");
        }

        [HttpPut]
        public ActionResult Edit(int id, Distribuitor distribuitorRequest)
        {
            try
            {
                
                if (ModelState.IsValid)
                {
                    Distribuitor distribuitor = ctx.Distribuitors
                   .Include("ContactInfo")
                    .SingleOrDefault(b => b.DistribuitorId.Equals(id));
                    if (TryUpdateModel(distribuitor))
                    {
                        distribuitor.Nume = distribuitorRequest.Nume;
                        distribuitor.ContactInfo.PhoneNumber = distribuitorRequest.ContactInfo.PhoneNumber;
                        distribuitor.ContactInfo.Adresa = distribuitorRequest.ContactInfo.Adresa;
                        distribuitor.ContactInfo.StartDay = distribuitorRequest.ContactInfo.StartDay;
                        distribuitor.ContactInfo.StartMonth = distribuitorRequest.ContactInfo.StartMonth;
                        distribuitor.ContactInfo.StartYear = distribuitorRequest.ContactInfo.StartYear;
                        ctx.SaveChanges();
                    }
                    return RedirectToAction("Index");
                }
                return View(distribuitorRequest);
            }
            catch (Exception e)
            {
                return View(distribuitorRequest);
            }
        }

    }
}