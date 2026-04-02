//#define _CRT_SECURE_NO_WARNINGS 1
//#include <bits/stdc++.h>
//#define ll long long
//#define ld long double
//#define ull unsigned long long
//#define endl '\n'
//using namespace std;
//map<int, int> mp;
//void solve()
//{
//    mp.clear();
//    ull n;
//    int a[100005];
//    cin >> n;
//    ull ans = 0;
//    if (n % 2 == 1)
//    {
//        for (int i = n; i >= 1; i-=2)
//        {
//            ans += i * (n - i + 1);
//        }
//    }
//    else
//    {
//        for (int i = n - 1; i >= 1; i -= 2)
//        {
//            ans += i * (n - i + 1);
//        }
//    }
//    for (int i = 1; i <= n; i++)
//    {
//        mp[a[i]]++;
//        cin >> a[i];
//    }
//    sort(a + 1, a + n + 1);
//    cout << ans << endl;
//}
//int main()
//{
//    ios::sync_with_stdio(false);
//    cin.tie(0);
//    ll T = 1;
//    cin >> T;
//    
//    while (T--)
//    {
//        solve();
//    }
//    return 0;
//}