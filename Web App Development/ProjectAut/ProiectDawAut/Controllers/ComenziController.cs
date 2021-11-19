using ProiectDawAut.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;
using System.Web.Mvc;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;
using Microsoft.AspNet.Identity;

namespace ProiectDawAut.Controllers
{
    public class ComenziController : Controller
    {
        // GET: Comenzi
        private ApplicationDbContext db = new ApplicationDbContext();
        [Authorize(Roles = "User,Admin")]
        [HttpGet]
        public ActionResult Index() 
        {
            List<Comenzi> comenzi = db.Comenzi.Include("User").ToList();
            ViewBag.Comenzi = comenzi;
            return View();
        }
        [Authorize(Roles = "User,Admin")]
        public ActionResult Details(int? id)
        {
            if (id.HasValue)
            {
                Comenzi comanda = db.Comenzi.Find(id);
                if(User.Identity.GetUserId() != comanda.UserId)
                    return HttpNotFound("Interzis!");
                if (comanda != null)
                {
                    return View(comanda);
                }
                return HttpNotFound("Couldn't find the order with id " + id.ToString() + "!");
            }
            return HttpNotFound("Missing order id parameter!");
        }

        [Authorize(Roles = "User,Admin")]
        [HttpGet]
        public ActionResult New()
        {
            Comenzi comanda = new Comenzi();
            comanda.BijuteriiList = GetAllBijuterii();
            comanda.Bijuterii = new List<Bijuterii>();
            return View(comanda);
        }
        [HttpPost]
        [Authorize(Roles = "User,Admin")]
        public ActionResult New(Comenzi comandaRequest)
        {
            comandaRequest.BijuteriiList = GetAllBijuterii();
            var currentUserId = User.Identity.GetUserId();
            var selectedBijuterii = comandaRequest.BijuteriiList.Where(b => b.Checked).ToList();
            try
            {
                if (ModelState.IsValid)
                {
                    comandaRequest.Bijuterii = new List<Bijuterii>();
                    for (int i = 0; i < selectedBijuterii.Count(); i++)
                    {

                        Bijuterii bijuterie = db.Bijuterii.Find(selectedBijuterii[i].Id);
                        comandaRequest.Bijuterii.Add(bijuterie);
                    }
                    comandaRequest.UserId = currentUserId;
                    db.Comenzi.Add(comandaRequest);
                    db.SaveChanges();
                    return RedirectToAction("Index");
                }
                return View(comandaRequest);
            }
            catch (Exception e)
            {
                var msg = e.Message;
                return View(comandaRequest);
            }
        }
        [HttpGet]
        [Authorize(Roles = "User,Admin")]
        public ActionResult Edit(int? id)
        {
            
            if (id.HasValue)
            {
                Comenzi comanda = db.Comenzi.Find(id);
                if (User.Identity.GetUserId() != comanda.UserId)
                    return HttpNotFound("Interzis!");
                comanda.BijuteriiList = GetAllBijuterii();    

                foreach (Bijuterii checkedBijuterii in comanda.Bijuterii)
                {
                    comanda.BijuteriiList.FirstOrDefault(g => g.Id == checkedBijuterii.IdBijuterie).Checked = true;
                }
                if (comanda == null)
                {
                    return HttpNotFound("Coludn't find the order with id " + id.ToString() + "!");
                }
                return View(comanda);
            }
            return HttpNotFound("Missing order id parameter!");
        }

        [HttpPut]
        [Authorize(Roles = "User,Admin")]
        public ActionResult Edit(int id, Comenzi comandaRequest)
        {
            comandaRequest.BijuteriiList = GetAllBijuterii();
            
            Comenzi comanda = db.Comenzi.Include("User")
                        .SingleOrDefault(b => b.IdComanda.Equals(id));

            var selectedBijuterii = comandaRequest.BijuteriiList.Where(b => b.Checked).ToList();
            try
            {
                if (ModelState.IsValid)
                {
                    if (TryUpdateModel(comanda))
                    {

                        comanda.Bijuterii.Clear();
                        comanda.Bijuterii = new List<Bijuterii>();
                        int total = 0;
                        for (int i = 0; i < selectedBijuterii.Count(); i++)
                        {
                            
                            Bijuterii bijuterie = db.Bijuterii.Find(selectedBijuterii[i].Id);
                            total = total + bijuterie.Pret;
                            comanda.Bijuterii.Add(bijuterie);
                        }
                        comanda.Total = total;
                        db.SaveChanges();
                    }
                    return RedirectToAction("Index");
                }
                return View(comandaRequest);
            }
            catch (Exception)
            {
                return View(comandaRequest);
            }
        }

        
        [Authorize(Roles = "User,Admin")]
        [HttpDelete]
        public ActionResult Delete(int id)
        {
            Comenzi comanda = db.Comenzi.Find(id);
            if (User.Identity.GetUserId() != comanda.UserId)
                return HttpNotFound("Interzis!");
            if (comanda != null)
            {
                db.Comenzi.Remove(comanda);
                db.SaveChanges();
                return RedirectToAction("Index");
            }
            return HttpNotFound("Couldn't find the order with id " + id.ToString() + "!");
        }
        [NonAction]
        public List<CheckBoxViewModel> GetAllBijuterii()
        {
            var checkboxList = new List<CheckBoxViewModel>();
            foreach (var bijuterie in db.Bijuterii.ToList())
            {
                checkboxList.Add(new CheckBoxViewModel
                {
                    Id = bijuterie.IdBijuterie,
                    Nume = bijuterie.Nume,
                    Checked = false
                });
            }
            return checkboxList;
        }
        
    }
}