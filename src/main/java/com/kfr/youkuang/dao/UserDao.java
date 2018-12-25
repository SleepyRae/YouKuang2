package com.kfr.youkuang.dao;

import com.kfr.youkuang.entity.User;
import com.kfr.youkuang.mapper.UserMapper;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Repository;

/**
 * @author WallfacerRZD
 * @date 2018/11/19 22:25
 */
@Repository
public class UserDao {
    private final UserMapper userMapper;

    @Autowired
    public UserDao(UserMapper userMapper) {
        this.userMapper = userMapper;
    }

    public User selectUserByUserName(final String userName) {
        return userMapper.selectUserByUserName(userName);
    }

    public User selectUserByUserID(final int userID) {
        return userMapper.selectUserByUserID(userID);
    }

    public void insertOneUser(final User user) {
        final String userName = user.getUserName();
        final String password = user.getPassword();
        System.out.println(userName);
        System.out.println(password);
        userMapper.insertOneUser(userName, password);
    }

}
