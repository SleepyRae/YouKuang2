package com.kfr.youkuang.action;

import com.kfr.youkuang.entity.AccountItem;
import com.kfr.youkuang.service.AccountItemService;

import jdk.nashorn.internal.runtime.logging.Logger;
import org.apache.ibatis.annotations.Param;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import javax.servlet.http.HttpServletRequest;
import java.util.List;


@RestController
    public class AccountItemController {
        private AccountItemService accountItemService;

        @Autowired
        public AccountItemController(AccountItemService accountItemService){
            this.accountItemService =accountItemService;
        }

        //	查询账本内所有账目
        @GetMapping("/account/{accountID}")
        public List<AccountItem> getAllItems(@PathVariable(name = "accountID") int accountID,
                                             HttpServletRequest request){
            return accountItemService.getAllItems(accountID,request);
        }

        //记一笔
        @PutMapping("/account/{accountID}")
        public void insert (@RequestBody AccountItem accountItem,
                            @PathVariable("accountID") int accountID,
                            HttpServletRequest request){
            accountItemService.insert(accountItem,accountID,request);
        }
        //修改账目
        @PatchMapping("/account/{accountID}")
        public void moodify(@RequestBody AccountItem accountItem,
                            @PathVariable("accountID") int accountID,
                            HttpServletRequest request){
        accountItemService.modify(accountItem, accountID, request);
    }
        //删除账目
        @DeleteMapping("/account/{accountID}")
        public void delete (@RequestBody AccountItem accountItem,
                            @PathVariable("accountID") int accountID,
                            HttpServletRequest request){
            accountItemService.delete(accountItem,accountID,request);
        }







}
