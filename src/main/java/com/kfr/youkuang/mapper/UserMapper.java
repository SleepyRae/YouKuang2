package com.kfr.youkuang.mapper;

import com.kfr.youkuang.entity.User;
import org.apache.ibatis.annotations.Insert;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;
import org.springframework.stereotype.Component;

/**
 * @author WallfacerRZD
 * @date 2018/11/9 12:13
 */
@Mapper
@Component
public interface UserMapper {
    //按姓名查找用户
    @Select("SELECT * FROM youkuang_user where userName = #{userName}")
    User selectUserByUserName(@Param("userName") final String userName);

    //按ID查找用户
    @Select("SELECT * FROM youkuang_user where userID = #{userID}")
    User selectUserByUserID(@Param("userID") final int userID);

    //插入新用户
    @Insert("INSERT INTO youkuang_user(userName, password) values(#{userName}, #{password})")
    void insertOneUser(@Param("userName") final String userName,
                       @Param("password") final String password);
}
