package main

import (
	"bytes"
	"database/sql"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"mime/multipart"
	"net/http"
	"strconv"
	"sync"
	"time"

	_ "github.com/go-sql-driver/mysql"
	"github.com/jasonlvhit/gocron"
)

//CategoryList : request category list
var CategoryList = []Category{
	//礦石
	Category{25, 1},
	//草木
	Category{25, 2},
	//種子/果實
	Category{25, 3},
	//皮
	Category{25, 4},
	//肉
	Category{25, 5},
	//血
	Category{25, 6},
	//海鮮
	Category{25, 7},
	//其他材料
	Category{25, 8},
	//黑石
	Category{30, 1},
	//改良
	Category{30, 2},
	//攻擊型靈藥
	Category{35, 1},
	//防禦型靈藥
	Category{35, 2},
	//功能型靈藥
	Category{35, 3},
	//料理
	Category{35, 4},
	//恢復劑
	Category{35, 5},
	//攻城相關
	Category{35, 6},
	//組合道具
	Category{35, 7},
	//其他消耗品
	Category{35, 8},
	//斧頭
	Category{40, 1},
	//吸管
	Category{40, 2},
	//屠刀
	Category{40, 3},
	//十字鎬
	Category{40, 4},
	//鋤頭
	Category{40, 5},
	//皮革刀
	Category{40, 6},
	//釣魚工具
	Category{40, 7},
	//火繩槍
	Category{40, 8},
	//煉金/料理工具
	Category{40, 9},
	//其他工具
	Category{40, 10},
}

//save the data
var items []Item

//Category object
type Category struct {
	mainCategory int
	subCategory  int
}

//Detail object
type Detail struct {
	PricePerOne     int    `json:"pricePerOne"`
	TotalTradeCount int64  `json:"totalTradeCount"`
	KeyType         int    `json:"keyType"`
	MainKey         int    `json:"mainKey"`
	SubKey          int    `json:"subKey"`
	Count           int    `json:"count"`
	Name            string `json:"name"`
	Grade           int    `json:"grade"`
	MainCategory    int    `json:"mainCategory"`
	SubCategory     int    `json:"subCategory"`
	ChooseKey       int    `json:"chooseKey"`
}

//MarketDetail object
type MarketDetail struct {
	DetailList []Detail `json:"detailList"`
	ResultCode int      `json:"resultCode"`
	ResultMsg  string   `json:"resultMsg"`
}

//Item object
type Item struct {
	MainKey       int    `json:"mainKey"`
	SumCount      int64  `json:"sumCount"`
	TotalSumCount int64  `json:"totalSumCount"`
	Name          string `json:"name"`
	Grade         int    `json:"grade"`
	MinPrice      int64  `json:"minPrice"`
}

//MarketItem object
type MarketItem struct {
	MarketList []Item `json:"marketList"`
	ResultCode int    `json:"resultCode"`
	ResultMsg  string `json:"resultMsg"`
}

func main() {
	//save records
	gocron.Every(1).Day().At("02:00").Do(getDataTask)
	gocron.Every(1).Day().At("08:00").Do(getDataTask)
	gocron.Every(1).Day().At("14:00").Do(getDataTask)
	gocron.Every(1).Day().At("20:00").Do(getDataTask)
	//save daily records
	gocron.Every(1).Day().At("23:55").Do(calAndSaveTradingVolume)

	_, nextTime := gocron.NextRun()
	fmt.Printf("First job will excute at %s\n", nextTime.Format("2006-01-02 15:04:05"))
	<-gocron.Start()
	getDataTask()
}

func getDataTask() {

	wg := sync.WaitGroup{}
	wg.Add(len(CategoryList))

	for _, category := range CategoryList {
		go getData(category, &wg)
	}
	wg.Wait()
	//after get data
	for _, item := range items {
		fmt.Printf("%d %d %d %s\n", item.MainKey, item.SumCount, item.TotalSumCount, item.Name)
	}
	fmt.Printf("%s succes, total data:%d\n", time.Now().Format("2006-01-02 15:04:05"), len(items))
	//show next time
	_, nextTime := gocron.NextRun()
	fmt.Printf("Next job will excute at %s\n", nextTime.Format("2006-01-02 15:04:05"))
	//save data to
	saveData(items)
	//clear the item array and wait for next time
	items = nil
}

func getData(category Category, wg *sync.WaitGroup) {
	url := "https://trade.tw.playblackdesert.com/Home/GetWorldMarketList"
	method := "POST"

	payload := &bytes.Buffer{}
	writer := multipart.NewWriter(payload)
	//Your Token after login
	_ = writer.WriteField("__RequestVerificationToken", "")
	_ = writer.WriteField("mainCategory", strconv.Itoa(category.mainCategory))
	_ = writer.WriteField("subCategory", strconv.Itoa(category.subCategory))
	err := writer.Close()
	if err != nil {
		fmt.Println(err)
	}

	client := &http.Client{}
	req, err := http.NewRequest(method, url, payload)

	if err != nil {
		fmt.Println(err)
	}
	req.Header.Add("User-Agent", " Mozilla/5.0 (iPad; CPU OS 11_0 like Mac OS X) AppleWebKit/604.1.34 (KHTML, like Gecko) Version/11.0 Mobile/15A5341f Safari/604.1")
	//Your Cookies after login*
	req.Header.Add("Cookie", "")
	req.Header.Set("Content-Type", writer.FormDataContentType())
	res, err := client.Do(req)
	defer res.Body.Close()
	body, _ := ioutil.ReadAll(res.Body)

	var data MarketItem
	json.Unmarshal([]byte(body), &data)
	for _, item := range data.MarketList {
		//save the data to item array
		item.TotalSumCount = getTotalSumCount(item.MainKey)
		items = append(items, item)
	}
	//Done a task
	defer wg.Done()
}

func getTotalSumCount(key int) int64 {
	url := "https://trade.tw.playblackdesert.com/Home/GetWorldMarketSubList"
	method := "POST"

	payload := &bytes.Buffer{}
	writer := multipart.NewWriter(payload)
	_ = writer.WriteField("__RequestVerificationToken", "8CoJOGqVRsURMG0X59QTwX9Gupo_vziH7sYD1YWmfjIIhvhphL8EyPaU2T-RoQbLhNFz0eaGSqK9gP1jnOHRCxOxgShuIy4l1dPxnCYHUGQ1")
	_ = writer.WriteField("mainKey", strconv.Itoa(key))
	_ = writer.WriteField("usingCleint", "0")
	err := writer.Close()
	if err != nil {
		fmt.Println(err)
	}

	client := &http.Client{}
	req, err := http.NewRequest(method, url, payload)

	if err != nil {
		fmt.Println(err)
	}
	req.Header.Add("User-Agent", " Mozilla/5.0 (iPad; CPU OS 11_0 like Mac OS X) AppleWebKit/604.1.34 (KHTML, like Gecko) Version/11.0 Mobile/15A5341f Safari/604.1")
	//Your Cookies after login*
	req.Header.Add("Cookie", "")

	req.Header.Set("Content-Type", writer.FormDataContentType())
	res, err := client.Do(req)
	defer res.Body.Close()
	body, _ := ioutil.ReadAll(res.Body)
	var data MarketDetail
	json.Unmarshal([]byte(body), &data)
	return data.DetailList[0].TotalTradeCount

}

func saveData(items []Item) {
	//Your database connection config
	db, err := sql.Open("mysql", "")
	if err != nil {
		fmt.Println(err.Error())
	}
	defer db.Close()

	//MainKey, Name, Grade
	stmtIns0, err := db.Prepare("INSERT IGNORE INTO bdo.trade_item VALUES( ?, ? ,?)")
	//Time, MainKey, SumCount, TotalSumCount, MinPrice
	stmtIns1, err := db.Prepare("INSERT INTO bdo.trade_record VALUES( ?, ? , ?, ?, ?)")
	if err != nil {
		fmt.Println(err.Error())
	}
	defer stmtIns0.Close()
	defer stmtIns1.Close()

	var rsra int64
	for _, item := range items {
		rs0, err := stmtIns0.Exec(item.MainKey, item.Name, item.Grade)
		fmt.Printf("%d %d %d %s\n", item.MainKey, item.SumCount, item.TotalSumCount, item.Name)
		rs1, err := stmtIns1.Exec(time.Now().Format("2006-01-02 15:04:05"), item.MainKey, item.SumCount, item.TotalSumCount, item.MinPrice)

		rsra0, err := rs0.RowsAffected()
		rsra1, err := rs1.RowsAffected()

		if err != nil {
			fmt.Println(err.Error())
		}

		rsra += (rsra0 + rsra1)
	}
	fmt.Printf("trade_record : inserted %d rows\n", rsra)
}

func calAndSaveTradingVolume() {
	//Your database connection config
	db, err := sql.Open("mysql", "")
	if err != nil {
		fmt.Println(err.Error())
	}
	defer db.Close()
	sql := "INSERT IGNORE INTO bdo.daily_record (MainKey,Date,TradingVolume) SELECT MainKey, DATE(Time) AS Date, MAX(TotalSumCount) - MIN(TotalSumCount) AS TradingVolume FROM bdo.trade_record GROUP BY MainKey, DATE(Time) ORDER BY MainKey ASC;"
	rs, err := db.Exec(sql)
	rsra, err := rs.RowsAffected()
	fmt.Printf("daily_record : inserted %d rows\n", rsra)
}
