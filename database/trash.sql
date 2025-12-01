-- phpMyAdmin SQL Dump
-- version 2.11.6
-- http://www.phpmyadmin.net
--
-- Host: localhost
-- Generation Time: Mar 03, 2024 at 04:08 PM
-- Server version: 5.0.51
-- PHP Version: 5.2.6

SET SQL_MODE="NO_AUTO_VALUE_ON_ZERO";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8 */;

--
-- Database: `trash`
--

-- --------------------------------------------------------

--
-- Table structure for table `admin`
--

CREATE TABLE `admin` (
  `username` varchar(20) NOT NULL,
  `password` varchar(20) NOT NULL,
  `mobile` bigint(20) NOT NULL,
  `email` varchar(40) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `admin`
--

INSERT INTO `admin` (`username`, `password`, `mobile`, `email`) VALUES
('admin', 'admin', 9975670006, '');

-- --------------------------------------------------------

--
-- Table structure for table `store_data`
--

CREATE TABLE `store_data` (
  `id` int(11) NOT NULL default '0',
  `otype` varchar(20) NOT NULL,
  `name` varchar(30) NOT NULL,
  `imgname` varchar(20) NOT NULL,
  `train_img` varchar(20) NOT NULL,
  `train_st` int(11) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `store_data`
--

INSERT INTO `store_data` (`id`, `otype`, `name`, `imgname`, `train_img`, `train_st`) VALUES
(1, 'Inorganic', 'bottle', 'image1.jpg', '', 1);

-- --------------------------------------------------------

--
-- Table structure for table `trash_alert`
--

CREATE TABLE `trash_alert` (
  `id` int(11) NOT NULL,
  `otype` varchar(20) NOT NULL,
  `name` varchar(30) NOT NULL,
  `dtime` timestamp NOT NULL default CURRENT_TIMESTAMP on update CURRENT_TIMESTAMP,
  `status` int(11) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data for table `trash_alert`
--

INSERT INTO `trash_alert` (`id`, `otype`, `name`, `dtime`, `status`) VALUES
(1, 'Inorganic', 'bottle', '2020-01-28 13:48:01', 0),
(2, 'Inorganic', 'bottle', '2020-01-28 13:48:06', 0),
(3, 'Inorganic', 'bottle', '2020-01-28 14:08:56', 0),
(4, 'Inorganic', 'bottle', '2020-01-28 14:08:59', 0),
(5, 'Inorganic', 'bottle', '2020-01-28 14:09:18', 0),
(6, 'Inorganic', 'bottle', '2020-01-28 14:09:19', 0),
(7, 'Inorganic', 'bottle', '2020-01-28 14:09:21', 0),
(8, 'Inorganic', 'bottle', '2020-01-28 14:09:22', 0),
(9, 'Inorganic', 'bottle', '2020-01-28 14:09:24', 0),
(10, 'Inorganic', 'bottle', '2020-01-28 14:09:25', 0),
(11, 'Inorganic', 'bottle', '2020-01-28 14:09:26', 0);
